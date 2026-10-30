"""Shared block-boundary rules for external and resumable prefill."""


def clamp_prefill_chunk_to_boundary(
    chunk_tokens: int, *, cache_tokens: int, block_size: int
) -> int:
    """Cap a non-empty chunk at the next cache block boundary.

    ``cache_tokens`` includes any reused prefix; ``block_size`` must be
    positive. Callers handle empty input and apply memory guards afterward.
    """
    next_boundary = ((cache_tokens // block_size) + 1) * block_size
    return max(1, min(chunk_tokens, next_boundary - cache_tokens))


def should_emit_prefill_boundary(
    *, total_tokens: int, block_size: int, last_emitted_tokens: int
) -> bool:
    """Identify a new, non-empty block boundary with a positive block size.

    Snapshot emission and updates to the last-emitted count belong to callers.
    """
    return (
        total_tokens > 0
        and total_tokens % block_size == 0
        and last_emitted_tokens < total_tokens
    )


def merge_short_prefill_chunk(
    chunk_tokens: int,
    *,
    remaining: int,
    block_size: int,
    step_size: int,
    min_chunk: int,
) -> int:
    """Grow a chunk so that no forward is shorter than ``min_chunk`` tokens.

    Applied after the boundary clamp. A short leading piece up to a block
    boundary (a prefix-cache resume that is not block aligned) runs together
    with the following block, and a short remainder after this chunk joins it.
    Models whose cost per forward barely depends on its length (one pass over
    SSD-offloaded experts per forward) gain the most. Chunks still end on a
    block boundary or at the end of the input, so boundary snapshots stay
    exact; only the snapshot at the skipped boundary is not taken.
    ``min_chunk <= 0`` disables the merge.
    """
    n = chunk_tokens
    if min_chunk <= 0 or n >= remaining:
        return n
    if n < min_chunk:
        n = min(remaining, n + (block_size if block_size > 0 else step_size))
    if 0 < remaining - n < min_chunk:
        n = remaining
    return n
