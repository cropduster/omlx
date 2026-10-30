"""P2' experiment: no prefill forward shorter than a minimum (DeepSeek V4.1 only, switchable)."""

from types import SimpleNamespace

import pytest

from omlx.prefill_boundaries import (
    clamp_prefill_chunk_to_boundary,
    merge_short_prefill_chunk,
)
from omlx.scheduler import Scheduler

BLOCK = STEP = 8192
MIN = 2048


def plan(total, cached, min_chunk=MIN, tail_at=None):
    """The forwards a prefill of ``total`` tokens after ``cached`` runs (clamp, merge, tail)."""
    chunks, pos = [], cached
    while pos < total:
        remaining = total - pos
        n = clamp_prefill_chunk_to_boundary(min(STEP, remaining), cache_tokens=pos, block_size=BLOCK)
        n = merge_short_prefill_chunk(n, remaining=remaining, block_size=BLOCK, step_size=STEP,
                                      min_chunk=min_chunk)
        if tail_at is not None and pos < tail_at:
            n = min(n, tail_at - pos)
        chunks.append(n)
        pos += n
    return chunks


@pytest.mark.parametrize(
    "total,cached,before,after",
    [
        # Fresh 9155-token prompt (measured 29.10.: 8192 + 963, the 963 cost ~19 s).
        (9155, 0, [8192, 963], [9155]),
        # Prefix resume at 32022 (+12.9k): 746 up to the boundary, then 8192 + 3999.
        (44958, 32022, [746, 8192, 3998], [8938, 3998]),
        # Aligned resume and long remainders are untouched.
        (24576, 8192, [8192, 8192], [8192, 8192]),
        (20000, 0, [8192, 8192, 3616], [8192, 8192, 3616]),
        # Short prompts stay one forward.
        (1500, 0, [1500], [1500]),
    ],
)
def test_forward_plans(total, cached, before, after):
    assert plan(total, cached, min_chunk=0) == before
    assert plan(total, cached) == after


def test_chunks_still_end_on_a_boundary_or_at_the_end():
    for cached in (0, 100, 8000, 8191, 8192, 30000):
        for total in (cached + 1, cached + 2047, cached + 2049, cached + 9000, cached + 30000):
            pos = cached
            for n in plan(total, cached):
                pos += n
                assert pos == total or pos % BLOCK == 0
            assert pos == total


def test_tail_position_is_still_honoured():
    # The scheduler stops once more at the tail position (prompt - 2).
    assert plan(9155, 0, tail_at=9153) == [9153, 2]


def test_merge_is_bounded():
    # At most one block plus a remainder below the minimum.
    for cached in range(0, 8192, 97):
        for n in plan(cached + 40000, cached):
            assert n < BLOCK + 2 * MIN


def test_disabled_merge_is_the_identity():
    for n, remaining in ((10, 50), (8192, 9000), (5000, 5000)):
        assert merge_short_prefill_chunk(n, remaining=remaining, block_size=BLOCK, step_size=STEP,
                                         min_chunk=0) == n


def _sched(model_type):
    return SimpleNamespace(model=SimpleNamespace(model_type=model_type))


def test_scheduler_gate(monkeypatch, tmp_path):
    switch = tmp_path / "merge.on"
    monkeypatch.setattr(Scheduler, "PREFILL_MERGE_SWITCH", str(switch))
    monkeypatch.delenv(Scheduler.PREFILL_MERGE_ENV, raising=False)
    v41 = _sched("deepseek_v41")
    assert Scheduler._prefill_merge_min_chunk(v41) == 0  # off by default
    switch.write_text("")
    assert Scheduler._prefill_merge_min_chunk(v41) == Scheduler.PREFILL_MERGE_MIN_TOKENS
    # Other models never merge, whatever the switch says.
    assert Scheduler._prefill_merge_min_chunk(_sched("qwen3_5_moe")) == 0
    monkeypatch.setenv(Scheduler.PREFILL_MERGE_ENV, "0")
    assert Scheduler._prefill_merge_min_chunk(v41) == 0
    monkeypatch.setenv(Scheduler.PREFILL_MERGE_ENV, "1024")
    assert Scheduler._prefill_merge_min_chunk(v41) == 1024
    assert Scheduler._merge_short_prefill_chunk(
        v41, 746, remaining=12936, block_size=BLOCK, step_size=STEP, request_id="r"
    ) == 746 + BLOCK
