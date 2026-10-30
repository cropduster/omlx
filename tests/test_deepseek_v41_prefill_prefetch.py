"""P5 experiment: next-layer speculative prefill reads (switchable, off by default)."""

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.deepseek_v41 import moe_offload
from omlx.patches.deepseek_v41.moe_offload import OffloadedExpert


def _layers(tmp_path, n_layers=4, fraction=0.25, seed=21):
    """A stacked affine checkpoint with ``n_layers`` offloaded layers and resident references."""
    from mlx.utils import tree_flatten

    from omlx.patches.deepseek_v41.config import ModelConfig
    from omlx.patches.deepseek_v41.language import Expert
    from omlx.patches.deepseek_v41.moe_offload import ExpertOffloadPlan
    from omlx.patches.deepseek_v41.quantization import QuantizedProjection

    mx.random.seed(seed)
    config = ModelConfig(dim=64, moe_inter_dim=64, n_layers=n_layers, n_routed_experts=8,
                         n_activated_experts=2)
    tensors, specs, refs = {}, {}, []
    for layer in range(n_layers):
        ref = Expert(config, True)
        prefix = f"language_model.layers.{layer}.ffn.experts"
        for proj in ("w1", "w3", "w2"):
            q = mx.quantize(mx.random.normal((8, 64, 64)).astype(mx.bfloat16) * 0.05,
                            bits=4, group_size=32, mode="affine")
            setattr(ref, proj, QuantizedProjection(q[0], q[1], 4, "affine", biases=q[2]))
            specs[f"{prefix}.{proj}"] = dict(bits=4, mode="affine")
        tensors.update({prefix + "." + k: v for k, v in tree_flatten(ref.parameters())})
        refs.append(ref)
    mx.save_safetensors(str(tmp_path / "model.safetensors"), tensors)
    mapping = {k: "model.safetensors" for k in tensors}
    raw = {"omlx_deepseek_v41": {"version": 1, "quantized_modules": specs}}
    plan = ExpertOffloadPlan(tmp_path, raw, mapping, config, fraction)
    disks = [OffloadedExpert(Expert(config, True), plan, f"language_model.layers.{i}.ffn.experts")
             for i in range(n_layers)]
    return refs, disks, plan


@pytest.fixture
def switches(tmp_path, monkeypatch):
    for env in (moe_offload.PREFETCH_ENV, moe_offload.RESIDENT_FIRST_ENV, moe_offload.STATS_ENV):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setattr(moe_offload, "PREFETCH_SWITCH", tmp_path / "pf.on")
    monkeypatch.setattr(moe_offload, "RESIDENT_FIRST_SWITCH", tmp_path / "rf.on")
    monkeypatch.setattr(moe_offload, "STATS_SWITCH", tmp_path / "st.on")
    # Synthetic chunks are tiny: let them count as full prefill chunks.
    monkeypatch.setattr(moe_offload, "PREFETCH_MIN_TOKENS", 1)


def _chunk(seed, n=24):
    rng = np.random.default_rng(seed)
    ids = sorted(rng.integers(0, 8, size=n).tolist())
    mx.random.seed(seed)
    x = mx.random.normal((n, 1, 64)).astype(mx.bfloat16)
    w = mx.array(rng.uniform(0.1, 1.0, size=n).astype(np.float32))
    return ids, mx.array(ids), x, w


def test_off_by_default(tmp_path, switches):
    _, disks, plan = _layers(tmp_path)
    try:
        assert moe_offload.prefetch_enabled() is False
        ids, idx, x, w = _chunk(1)
        mx.eval(disks[0](x, idx, w, sorted_indices=True))
        assert disks[1].slots._ahead == {}
    finally:
        plan.close()


@pytest.mark.parametrize("resident_first", [False, True])
def test_next_layer_reads_start_and_are_consumed(tmp_path, switches, monkeypatch, resident_first):
    monkeypatch.setenv(moe_offload.PREFETCH_ENV, "1")
    monkeypatch.setenv(moe_offload.RESIDENT_FIRST_ENV, "1" if resident_first else "0")
    refs, disks, plan = _layers(tmp_path)
    try:
        ids, idx, x, w = _chunk(2)
        mx.eval(disks[0](x, idx, w, sorted_indices=True))
        # Layer 1 is in the lower half (4 layers -> split 2): its lowest
        # non-resident experts are already being read.
        ahead = disks[1].slots._ahead
        assert 0 < len(ahead) <= plan.capacity
        assert sorted(ahead) == sorted(ahead)[: len(ahead)] and min(ahead) == 0
        out = disks[1](x, idx, w, sorted_indices=True)
        expected = refs[1](x, idx, w, sorted_indices=True)
        mx.eval(out, expected)
        np.testing.assert_allclose(out.astype(mx.float32), expected.astype(mx.float32),
                                   rtol=0.01, atol=0.002)
        assert disks[1].slots._ahead == {}
        # Layer 1 -> 2 crosses into the upper half: no speculation there.
        assert disks[2].slots._ahead == {}
    finally:
        plan.close()


def test_outputs_and_reads_match_without_prefetch(tmp_path, switches, monkeypatch):
    results = {}
    for on in ("0", "1"):
        monkeypatch.setenv(moe_offload.PREFETCH_ENV, on)
        sub = tmp_path / on
        sub.mkdir()
        _, disks, plan = _layers(sub)
        try:
            outs = []
            for k in range(3):
                ids, idx, x, w = _chunk(10 + k)
                h = x
                for disk in disks:
                    h = disk(h, idx, w, sorted_indices=True)
                    mx.eval(h)
                    h = h.reshape(-1, 1, 64)
                outs.append(np.array(h.astype(mx.float32)))
            results[on] = (outs, [d.slots.misses for d in disks])
        finally:
            plan.close()
    for a, b in zip(results["0"][0], results["1"][0]):
        np.testing.assert_array_equal(a, b)
    # Prefetched experts are counted as misses when consumed, exactly as before.
    assert results["0"][1] == results["1"][1]


def test_decode_drops_stale_speculation(tmp_path, switches, monkeypatch):
    monkeypatch.setenv(moe_offload.PREFETCH_ENV, "1")
    _, disks, plan = _layers(tmp_path)
    try:
        ids, idx, x, w = _chunk(3)
        mx.eval(disks[0](x, idx, w, sorted_indices=True))
        assert disks[1].slots._ahead
        di = mx.array([[1, 4]])
        mx.eval(disks[1](mx.random.normal((1, 1, 1, 64)).astype(mx.bfloat16), di, mx.ones(di.shape) / 2))
        assert disks[1].slots._ahead == {}
    finally:
        plan.close()


def test_wasted_reads_are_reported(tmp_path, switches, monkeypatch):
    monkeypatch.setenv(moe_offload.PREFETCH_ENV, "1")
    _, disks, plan = _layers(tmp_path)
    lines = []
    stats = moe_offload.OffloadStats(4, 2, emit=lines.append, watchdog=False)
    for d in disks:
        d.slots.stats = stats
    try:
        # Layer 0 routes everything; layer 1 only to experts 6 and 7, so the
        # speculative reads of its lowest ids go unused.
        ids, idx, x, w = _chunk(4)
        mx.eval(disks[0](x, idx, w, sorted_indices=True))
        few = mx.array([6] * 12 + [7] * 12)
        mx.eval(disks[1](x, few, w, sorted_indices=True))
        mx.eval(disks[2](x, few, w, sorted_indices=True))
        mx.eval(disks[3](x, few, w, sorted_indices=True))
        assert "wasted=" in lines[-1] and int(lines[-1].split("wasted=")[1].split()[0]) > 0
    finally:
        plan.close()
