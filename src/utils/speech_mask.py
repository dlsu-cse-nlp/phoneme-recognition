import numpy as np


def speech_mask(
    wave,
    n_frames,
    hop=640,
    frac=0.5,
    min_range_db=12.0,
    max_gap=6,
    min_len=3,
    pad=2,
):
    # Energy-based per-frame speech / non-speech mask on the MD&D frame grid.
    wave = np.asarray(wave, dtype=np.float64).reshape(-1)
    n = max(int(np.ceil(len(wave) / hop)), 1) * hop
    w = np.pad(wave, (0, n - len(wave)))
    db = 20 * np.log10(np.sqrt((w.reshape(-1, hop) ** 2).mean(1)) + 1e-9)

    floor, peak = np.percentile(db, 10), np.percentile(db, 95)
    if peak - floor < min_range_db:
        return np.ones(n_frames, dtype=bool)

    m = db > floor + frac * (peak - floor)

    idx = np.flatnonzero(m)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap + 1:
            m[a:b] = True

    out = np.zeros_like(m)
    i = 0
    while i < len(m):
        if m[i]:
            j = i
            while j < len(m) and m[j]:
                j += 1
            if j - i >= min_len:
                out[i:j] = True
            i = j
        else:
            i += 1

    out = np.convolve(out.astype(float), np.ones(2 * pad + 1), "same") > 0
    out = out[:n_frames]
    return np.pad(out, (0, n_frames - len(out)))
