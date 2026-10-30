"""P1a experiment: resident-first prefill order (switchable, off by default)."""

import mlx.core as mx
import numpy as np
import pytest
from test_deepseek_v41_moe_offload import _synthetic_affine_experts

from omlx.patches.deepseek_v41 import moe_offload
from omlx.patches.deepseek_v41.language import Expert


@pytest.fixture
def switch(tmp_path, monkeypatch):
    monkeypatch.delenv(moe_offload.RESIDENT_FIRST_ENV, raising=False)
    monkeypatch.delenv(moe_offload.STATS_ENV, raising=False)
    monkeypatch.setattr(moe_offload, "RESIDENT_FIRST_SWITCH", tmp_path / "rf.on")
    monkeypatch.setattr(moe_offload, "STATS_SWITCH", tmp_path / "stats.on")
    return tmp_path / "rf.on"


def _routes(seed=5, n=15):
    rng = np.random.default_rng(seed)
    ids = sorted(rng.integers(0, 8, size=n).tolist())
    idx = mx.array(ids)
    mx.random.seed(seed)
    x = mx.random.normal((len(ids), 1, 64)).astype(mx.bfloat16)
    weights = mx.array(rng.uniform(0.1, 1.0, size=len(ids)).astype(np.float32))
    return ids, idx, x, weights


def _build(tmp_path, name, enabled, monkeypatch):
    if enabled:
        monkeypatch.setenv(moe_offload.RESIDENT_FIRST_ENV, "1")
    else:
        monkeypatch.setenv(moe_offload.RESIDENT_FIRST_ENV, "0")
    sub = tmp_path / name
    sub.mkdir()
    return _synthetic_affine_experts(sub, 0.25)


def test_switch_defaults_off_and_is_read_per_load(tmp_path, switch, monkeypatch):
    assert moe_offload.resident_first_enabled() is False
    switch.write_text("")
    assert moe_offload.resident_first_enabled() is True
    monkeypatch.setenv(moe_offload.RESIDENT_FIRST_ENV, "0")
    assert moe_offload.resident_first_enabled() is False
    _, disk, plan = _build(tmp_path, "a", False, monkeypatch)
    try:
        assert disk._resident_first is False
    finally:
        plan.close()


def test_second_sweep_reads_only_what_is_not_resident(tmp_path, switch, monkeypatch):
    reference, disk, plan = _build(tmp_path, "rf", True, monkeypatch)
    try:
        ids = list(range(8)) * 2
        ids.sort()
        idx = mx.array(ids)
        x = mx.random.normal((len(ids), 1, 64)).astype(mx.bfloat16)
        weights = mx.ones(idx.shape) / 2
        first = disk(x, idx, weights, sorted_indices=True)
        mx.eval(first)
        assert disk.slots.misses == 8 and disk.slots.hits == 0
        resident = set(disk.slots.slot_of)
        second = disk(x, idx, weights, sorted_indices=True)
        expected = reference(x, idx, weights, sorted_indices=True)
        mx.eval(second, expected)
        # Two slots stayed resident: the second sweep reads 8 - 2 experts, not 8.
        assert disk.slots.misses == 8 + 6
        assert disk.slots.hits == len(resident) == 2
        assert disk.slots._ahead == {}
        for out in (first, second):
            np.testing.assert_allclose(
                out.astype(mx.float32), expected.astype(mx.float32), rtol=0.01, atol=0.002
            )
    finally:
        plan.close()


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_resident_first_matches_the_sorted_path(tmp_path, switch, monkeypatch, seed):
    """Same outputs as the plain sorted path over several forwards, row by row."""
    results = {}
    for enabled in (False, True):
        _, disk, plan = _build(tmp_path, f"s{seed}{enabled}", enabled, monkeypatch)
        try:
            outs = []
            for k in range(3):
                ids, idx, x, weights = _routes(seed * 10 + k)
                out = disk(x, idx, weights, sorted_indices=True)
                mx.eval(out)
                outs.append(np.array(out.astype(mx.float32)))
            results[enabled] = (outs, disk.slots.misses)
        finally:
            plan.close()
    for a, b in zip(results[False][0], results[True][0]):
        np.testing.assert_allclose(a, b, rtol=0.01, atol=0.002)
    # Never more reads than the plain sorted path.
    assert results[True][1] <= results[False][1]


def test_groups_never_exceed_capacity_and_cover_every_route(tmp_path, switch, monkeypatch):
    _, disk, plan = _build(tmp_path, "g", True, monkeypatch)
    calls = []
    original = Expert.__call__

    def counted(self, x, indices=None, *args, **kwargs):
        if self is disk.slots.expert and kwargs.get("sorted_indices"):
            calls.append(np.array(indices).tolist())
        return original(self, x, indices, *args, **kwargs)

    monkeypatch.setattr(Expert, "__call__", counted)
    try:
        for seed in (7, 8):
            ids, idx, x, weights = _routes(seed, n=24)
            calls.clear()
            mx.eval(disk(x, idx, weights, sorted_indices=True))
            assert sum(len(c) for c in calls) == len(ids)
            assert all(len(set(c)) <= plan.capacity for c in calls)
            assert all(c == sorted(c) for c in calls)
    finally:
        plan.close()


def test_resident_first_counts_hits_in_statistics(tmp_path, switch, monkeypatch):
    _, disk, plan = _build(tmp_path, "st", True, monkeypatch)
    lines = []
    disk.slots.stats = moe_offload.OffloadStats(1, 2, emit=lines.append, watchdog=False)
    try:
        ids = sorted(list(range(8)) * 2)
        idx = mx.array(ids)
        x = mx.random.normal((len(ids), 1, 64)).astype(mx.bfloat16)
        weights = mx.ones(idx.shape) / 2
        mx.eval(disk(x, idx, weights, sorted_indices=True))
        mx.eval(disk(x, idx, weights, sorted_indices=True))
        assert "hits=2 misses=6" in lines[1]
        assert "chunks=4" in lines[1]  # one resident group + three miss groups
    finally:
        plan.close()
