"""V4.1 expert-offload statistics: off by default, exact counters when on."""

import re

import mlx.core as mx
import numpy as np
import pytest
from test_deepseek_v41_moe_offload import _synthetic_affine_experts

from omlx.patches.deepseek_v41 import moe_offload
from omlx.patches.deepseek_v41.moe_offload import OffloadStats


def _fields(line):
    return dict(re.findall(r"(\w+)=([^\s]+)", line))


@pytest.fixture
def stats_off(tmp_path, monkeypatch):
    monkeypatch.delenv(moe_offload.STATS_ENV, raising=False)
    monkeypatch.setattr(moe_offload, "STATS_SWITCH", tmp_path / "absent.on")


@pytest.fixture
def stats_on(tmp_path, monkeypatch):
    monkeypatch.setenv(moe_offload.STATS_ENV, "1")
    monkeypatch.setattr(moe_offload, "STATS_SWITCH", tmp_path / "absent.on")


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def _collector(n_layers=4, top_k=2, idle_s=2.0):
    clock, lines = FakeClock(), []
    stats = OffloadStats(n_layers, top_k, clock=clock, emit=lines.append, idle_s=idle_s, watchdog=False)
    return stats, clock, lines


def _forward(stats, clock, phase, *, misses=1, hits=0, routes=4, nbytes=10, dt=0.1):
    for layer in range(stats.n_layers):
        stats.record(phase, layer, routes=routes, distinct=hits + misses, hits=hits, misses=misses,
                     nbytes=nbytes, wait=0.01, sync=0.0, ensure=0.02, tolist=0.005)
        stats.layer_done(phase, layer)
    clock.now += dt


# --- switch ---------------------------------------------------------------


def test_switch_is_off_by_default(tmp_path, stats_off):
    _, disk, plan = _synthetic_affine_experts(tmp_path, 0.25)
    try:
        assert moe_offload.stats_enabled() is False
        assert disk.slots.stats is None
        assert getattr(plan, "_offload_stats") is None
    finally:
        plan.close()


def test_switch_file_and_environment(tmp_path, monkeypatch):
    switch = tmp_path / "dsv41-offload-stats.on"
    monkeypatch.setattr(moe_offload, "STATS_SWITCH", switch)
    monkeypatch.delenv(moe_offload.STATS_ENV, raising=False)
    assert moe_offload.stats_enabled() is False
    switch.write_text("")
    assert moe_offload.stats_enabled() is True
    # An explicit "0" in the environment wins over the switch file.
    monkeypatch.setenv(moe_offload.STATS_ENV, "0")
    assert moe_offload.stats_enabled() is False
    monkeypatch.setenv(moe_offload.STATS_ENV, "1")
    switch.unlink()
    assert moe_offload.stats_enabled() is True


def test_layer_index_from_prefix():
    assert moe_offload._layer_index("language_model.layers.37.ffn.experts") == 37
    assert moe_offload._layer_index("something.else") == -1


# --- collector --------------------------------------------------------------


def test_prefill_forward_is_one_line_with_layer_halves():
    stats, clock, lines = _collector()
    _forward(stats, clock, "prefill", misses=3, hits=1, routes=8, nbytes=2_000_000_000)
    assert len(lines) == 1
    f = _fields(lines[0])
    assert lines[0].startswith("DSV41-OFFLOAD prefill")
    assert f["req"] == "0" and f["fwd"] == "1" and f["chunks"] == "4"
    assert f["misses"] == "12" and f["hits"] == "4" and f["routes"] == "32"
    assert f["lo_misses"] == "6" and f["hi_misses"] == "6" and f["hi_routes"] == "16"
    assert float(f["read_gb"]) == pytest.approx(8.0)
    _forward(stats, clock, "prefill")
    assert _fields(lines[1])["fwd"] == "2"


def test_decode_windows_and_rates():
    stats, clock, lines = _collector(n_layers=4, top_k=2)
    for _ in range(64):
        # 3 verified rows x top-2 = 6 routes per layer per step.
        _forward(stats, clock, "decode", misses=2, routes=6, nbytes=1_000_000, dt=0.2)
    assert [_fields(line)["win"] for line in lines] == ["1-16", "17-64"]
    first = _fields(lines[0])
    assert first["steps"] == "16" and first["reason"] == "window"
    assert float(first["rows_per_step"]) == pytest.approx(3.0)
    assert float(first["misses_per_step"]) == pytest.approx(8.0)
    assert float(first["miss_per_kroute"]) == pytest.approx(1000 * 2 / 6, rel=1e-3)
    assert float(first["read_mb"]) == pytest.approx(64.0)
    assert float(first["tolist_ms_per_step"]) == pytest.approx(20.0)
    for _ in range(256):
        _forward(stats, clock, "decode")
    assert _fields(lines[-1])["win"] == "65-320"


def test_idle_flushes_partial_window_and_starts_a_request():
    stats, clock, lines = _collector(idle_s=2.0)
    for _ in range(5):
        _forward(stats, clock, "decode", dt=0.2)
    assert lines == []
    clock.now += 1.0
    stats.poll()
    assert lines == []  # not idle long enough yet
    clock.now += 2.0
    stats.poll()
    f = _fields(lines[0])
    assert f["win"] == "1-5" and f["reason"] == "idle" and f["req"] == "0"
    # Wall time ends at the last event, not at the poll.
    assert float(f["wall_ms_per_step"]) == pytest.approx(1000 * (0.8 + 0.0) / 5, rel=0.3)
    _forward(stats, clock, "decode")
    stats.poll()
    clock.now += 5
    stats.poll()
    assert _fields(lines[-1])["req"] == "1" and _fields(lines[-1])["win"] == "1-1"


def test_prefill_after_decode_closes_the_turn():
    stats, clock, lines = _collector()
    _forward(stats, clock, "prefill")
    for _ in range(20):
        _forward(stats, clock, "decode")
    _forward(stats, clock, "prefill")
    kinds = [(line.split()[1], _fields(line)["req"]) for line in lines]
    assert kinds == [("prefill", "0"), ("decode", "0"), ("decode", "0"), ("prefill", "1")]
    assert _fields(lines[2])["win"] == "17-20" and _fields(lines[2])["reason"] == "prefill"
    assert _fields(lines[3])["fwd"] == "1"


def test_unfinished_decode_layers_do_not_count_a_step():
    stats, clock, lines = _collector(n_layers=4)
    stats.record("decode", 0, routes=2, distinct=1, hits=1, misses=0, nbytes=0,
                 wait=0, sync=0, ensure=0)
    stats.layer_done("decode", 0)
    assert stats.decode_steps == 0


# --- integration with the offloaded expert ----------------------------------


def _attach(disk, n_layers=1, top_k=2):
    lines = []
    disk.slots.stats = OffloadStats(n_layers, top_k, emit=lines.append, watchdog=False)
    return lines


def test_sorted_sweep_reports_the_slot_counters(tmp_path, stats_on):
    _, disk, plan = _synthetic_affine_experts(tmp_path, 0.25)
    try:
        assert isinstance(plan._offload_stats, OffloadStats)
        plan._offload_stats.close()
        lines = _attach(disk)
        ids = sorted([0, 0, 0, 1, 2, 2, 3, 4, 4, 4, 4, 5, 6, 7, 7])
        idx = mx.array(ids)
        x = mx.random.normal((len(ids), 1, 64)).astype(mx.bfloat16)
        weights = mx.ones(idx.shape) / 2
        mx.eval(disk(x, idx, weights, sorted_indices=True))
        mx.eval(disk(x, idx, weights, sorted_indices=True))
        assert [line.split()[1] for line in lines] == ["prefill", "prefill"]
        first, second = (_fields(line) for line in lines)
        # Four chunks of two experts, eight misses; the second sweep thrashes.
        assert first["chunks"] == "4" and first["misses"] == "8" and first["hits"] == "0"
        assert first["routes"] == "15" and first["distinct"] == "8"
        assert second["misses"] == "8" and second["fwd"] == "2"
        assert int(first["read_bytes"]) + int(second["read_bytes"]) == disk.slots.fetched_bytes
        assert disk.slots.misses == 16
    finally:
        plan.close()


def test_routed_decode_reports_windows(tmp_path, stats_on):
    reference, disk, plan = _synthetic_affine_experts(tmp_path, 0.25)
    try:
        plan._offload_stats.close()
        lines = _attach(disk)
        rng = np.random.default_rng(3)
        for _ in range(16):
            idx = mx.array(rng.choice(8, size=(1, 2), replace=False))
            x = mx.random.normal((1, 1, 1, 64)).astype(mx.bfloat16)
            w = mx.ones(idx.shape) / 2
            out = disk(x, idx, w)
            mx.eval(out, reference(x, idx, w))
        assert len(lines) == 1
        f = _fields(lines[0])
        assert f["win"] == "1-16" and f["steps"] == "16"
        assert float(f["rows_per_step"]) == pytest.approx(1.0)
        assert int(f["misses"]) == disk.slots.misses
        assert int(f["hits"]) == disk.slots.hits
    finally:
        plan.close()


def test_outputs_are_unchanged_with_statistics_on(tmp_path, monkeypatch, stats_off):
    ids = sorted([0, 1, 1, 2, 3, 5, 5, 6, 7])
    idx = mx.array(ids)
    mx.random.seed(9)
    x = mx.random.normal((len(ids), 1, 64)).astype(mx.bfloat16)
    weights = mx.ones(idx.shape) / 2
    results = []
    for enabled in (False, True):
        if enabled:
            monkeypatch.setenv(moe_offload.STATS_ENV, "1")
        sub = tmp_path / str(enabled)
        sub.mkdir()
        _, disk, plan = _synthetic_affine_experts(sub, 0.25)
        try:
            out = disk(x, idx, weights, sorted_indices=True)
            mx.eval(out)
            results.append((np.array(out.astype(mx.float32)), disk.slots.misses, disk.slots.hits))
        finally:
            plan.close()
    np.testing.assert_array_equal(results[0][0], results[1][0])
    assert results[0][1:] == results[1][1:]
