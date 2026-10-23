import argparse
import shutil
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

RANDOM_STATE = 339

INPUT_MANIFEST = Path("data/fleurs/train_manifest.json")
OUTPUT_MANIFEST = Path("data/concat/train_manifest.json")
OUTPUT_AUDIO_DIR = Path("data/concat/audio/train")

SAMPLE_RATE = 16000  # FLEURS audio is 16 kHz mono
GAP_SECONDS = 0.1  # silence inserted between concatenated utterances
BIN_WIDTH = 2.0  # seconds, same buckets as the EDA notebook
FLAT_START = 8.0  # bins from here up to MAX_DURATION are flattened
MAX_DURATION = 40.0  # never build a sample longer than this (original max ~37s)
MAX_PARTS = 4  # max utterances glued into one sample
ATTEMPTS_PER_BIN = 200  # random tries to build a sample for a given bin


def load_manifest(path):
    return pd.read_json(path, orient="records", lines=True)


def _plan_for_level(
    durations,
    texts,
    level,
    *,
    gap,
    bin_width,
    flat_start,
    max_duration,
    max_parts,
    attempts,
    seed,
):
    """Try to lift every bin in the flat region up to ``level`` utterances.

    Only bins holding more than ``level`` utterances may donate sources, so
    natural short/mid-length data is never pushed below the target level.
    Returns ``(groups, success)``.
    """
    rng = np.random.default_rng(seed)
    n = len(durations)
    n_bins = int(np.ceil(max_duration / bin_width))
    first_bin = int(flat_start // bin_width)
    min_dur = durations.min()

    bin_idx = np.minimum((durations // bin_width).astype(int), n_bins - 1)
    counts = np.bincount(bin_idx, minlength=n_bins).astype(float)
    pool = np.ones(n, dtype=bool)  # utterances still available as sources
    same_text = {}  # text -> indices, for the "no repeated sentence" rule
    for i, t in enumerate(texts):
        same_text.setdefault(t, []).append(i)

    def try_build(b):
        """Try to build one sample whose duration falls in bin ``b``."""
        lo, hi = b * bin_width, (b + 1) * bin_width
        mid = (lo + hi) / 2
        # Fewer parts is more natural, so favour small k; skip k whose parts
        # would be shorter than the shortest real utterance.
        ks = [k for k in range(2, max_parts + 1) if mid / k >= min_dur]
        if not ks:
            return None
        k_weights = np.array([1.0 / k for k in ks])
        k_weights /= k_weights.sum()

        for _ in range(attempts):
            k = int(rng.choice(ks, p=k_weights))
            total_gap = gap * (k - 1)
            group, total = [], 0.0
            local_dec = np.zeros(n_bins)
            blocked = np.zeros(n, dtype=bool)
            ok = True
            for part in range(k):
                joins_left = k - 1 - part
                # A source's bin must stay above ``level`` even after this sample.
                donor_ok = (counts - local_dec)[bin_idx] > level
                cand = pool & ~blocked & donor_ok
                if part < k - 1:
                    # Leave room for the remaining parts (>= min_dur each).
                    cand &= durations <= hi - total_gap - total - joins_left * min_dur
                else:
                    cand &= (durations + total + total_gap >= lo) & (
                        durations + total + total_gap < hi
                    )
                idx = np.flatnonzero(cand)
                if len(idx) == 0:
                    ok = False
                    break
                # Prefer sources from the most over-populated bins.
                w = (counts - local_dec)[bin_idx[idx]] - level
                pick = int(rng.choice(idx, p=w / w.sum()))
                group.append(pick)
                total += durations[pick]
                local_dec[bin_idx[pick]] += 1
                blocked[pick] = True
                # Never put the same sentence twice in one sample.
                blocked[same_text[texts[pick]]] = True
            if ok:
                rng.shuffle(group)
                return group
        return None

    groups = []
    while True:
        region = counts[first_bin:]
        deficient = np.flatnonzero(region < level)
        if len(deficient) == 0:
            return groups, True
        # Visit the emptiest bins first until one can be filled.
        order = deficient[
            np.argsort(region[deficient] + rng.random(len(deficient)) * 0.1)
        ]
        built = False
        for off in order:
            b = first_bin + int(off)
            group = try_build(b)
            if group is None:
                continue
            pool[group] = False
            np.subtract.at(counts, bin_idx[group], 1)
            counts[b] += 1
            groups.append([int(i) for i in group])
            built = True
            break
        if not built:
            return groups, False


def plan_concatenations(
    durations,
    texts,
    *,
    level=None,
    gap=GAP_SECONDS,
    bin_width=BIN_WIDTH,
    flat_start=FLAT_START,
    max_duration=MAX_DURATION,
    max_parts=MAX_PARTS,
    attempts=ATTEMPTS_PER_BIN,
    seed=RANDOM_STATE,
):
    """Decide which utterances to concatenate. Returns a list of index lists.

    Each inner list holds row indices (in playback order) of one concatenated
    sample. Rows not appearing in any list stay as individual samples.

    ``level`` is the number of utterances each duration bin in
    ``[flat_start, max_duration)`` should reach. If None, the highest level
    that can actually be reached is found by binary search.
    """
    durations = np.asarray(durations, dtype=float)
    texts = list(texts)
    kwargs = dict(
        gap=gap,
        bin_width=bin_width,
        flat_start=flat_start,
        max_duration=max_duration,
        max_parts=max_parts,
        attempts=attempts,
        seed=seed,
    )
    if level is not None:
        groups, ok = _plan_for_level(durations, texts, level, **kwargs)
        if not ok:
            print(f"Warning: could not reach level {level}; using best effort.")
        return groups

    lo_level, hi_level = 1, int(np.bincount((durations // bin_width).astype(int)).max())
    best = []
    while lo_level <= hi_level:
        mid = (lo_level + hi_level) // 2
        groups, ok = _plan_for_level(durations, texts, mid, **kwargs)
        if ok:
            best, lo_level = groups, mid + 1
            chosen = mid
        else:
            hi_level = mid - 1
    print(f"Auto-selected target of {chosen} utterances per duration bin.")
    return best


def read_audio(path, sample_rate=SAMPLE_RATE):
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
    return audio


def print_distribution(title, durations, text_lens, bin_width=BIN_WIDTH):
    durations = np.asarray(durations)
    text_lens = np.asarray(text_lens)
    edges = np.arange(0, durations.max() + bin_width, bin_width)
    counts, _ = np.histogram(durations, bins=edges)
    print(f"\n{title}: {len(durations)} utterances, {durations.sum() / 3600:.2f} h")
    print(
        f"  duration  mean={durations.mean():.1f}s  median={np.median(durations):.1f}s"
        f"  p95={np.percentile(durations, 95):.1f}s  max={durations.max():.1f}s"
    )
    print(
        f"  text len  mean={text_lens.mean():.0f}  median={np.median(text_lens):.0f}"
        f"  p95={np.percentile(text_lens, 95):.0f}  max={text_lens.max()}"
    )
    scale = max(counts.max(), 1)
    for lo, c in zip(edges[:-1], counts):
        bar = "#" * int(40 * c / scale)
        print(f"  {lo:4.0f}-{lo + bin_width:<4.0f}s {c:5d} {bar}")


def main():
    parser = argparse.ArgumentParser(
        description="Build a length-balanced train manifest by concatenating utterances."
    )
    parser.add_argument("--input", type=Path, default=INPUT_MANIFEST)
    parser.add_argument("--output", type=Path, default=OUTPUT_MANIFEST)
    parser.add_argument("--audio-dir", type=Path, default=OUTPUT_AUDIO_DIR)
    parser.add_argument("--gap", type=float, default=GAP_SECONDS)
    parser.add_argument("--flat-start", type=float, default=FLAT_START)
    parser.add_argument("--max-duration", type=float, default=MAX_DURATION)
    parser.add_argument("--max-parts", type=int, default=MAX_PARTS)
    parser.add_argument(
        "--level",
        type=int,
        default=None,
        help="utterances per duration bin to aim for (default: auto)",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--dry-run", action="store_true", help="plan and print stats, write nothing"
    )
    args = parser.parse_args()

    df = load_manifest(args.input)
    texts = df["text"].astype(str).tolist()
    durations = df["duration"].to_numpy(dtype=float)
    paths = df["audio_filepath"].tolist()

    groups = plan_concatenations(
        durations,
        texts,
        gap=args.gap,
        flat_start=args.flat_start,
        max_duration=args.max_duration,
        max_parts=args.max_parts,
        level=args.level,
        seed=args.seed,
    )
    used = {i for g in groups for i in g}
    singles = [i for i in range(len(df)) if i not in used]
    print(
        f"Planned {len(groups)} concatenated samples from {len(used)} utterances; "
        f"{len(singles)} utterances kept as-is."
    )

    # Expected stats from the plan (actual durations are measured when writing).
    new_durations = [durations[i] for i in singles] + [
        sum(durations[i] for i in g) + args.gap * (len(g) - 1) for g in groups
    ]
    new_text_lens = [len(texts[i]) for i in singles] + [
        len(" ".join(texts[i] for i in g)) for g in groups
    ]
    print_distribution("BEFORE", durations, [len(t) for t in texts])
    print_distribution("AFTER (planned)", new_durations, new_text_lens)

    if args.dry_run:
        return

    args.audio_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    records = []
    silence = np.zeros(int(round(args.gap * SAMPLE_RATE)), dtype=np.float32)

    # 1) Concatenated samples
    for n, group in enumerate(tqdm(groups, desc="Concatenating")):
        pieces = []
        for j, i in enumerate(group):
            if j > 0:
                pieces.append(silence)
            pieces.append(read_audio(paths[i]))
        audio = np.concatenate(pieces)
        out_path = args.audio_dir / f"concat_{n:05d}.wav"
        sf.write(str(out_path), audio, SAMPLE_RATE, subtype="PCM_16")
        records.append(
            {
                "audio_filepath": out_path.as_posix(),
                "text": " ".join(texts[i] for i in group),
                "duration": round(len(audio) / SAMPLE_RATE, 2),
            }
        )

    # 2) Untouched samples, copied so the whole training set lives in one folder
    for i in tqdm(singles, desc="Copying"):
        src = Path(paths[i])
        dst = args.audio_dir / src.name
        shutil.copy2(src, dst)
        records.append(
            {
                "audio_filepath": dst.as_posix(),
                "text": texts[i],
                "duration": float(durations[i]),
            }
        )

    out_df = pd.DataFrame(records).sample(frac=1, random_state=args.seed)
    out_df.to_json(args.output, orient="records", lines=True, default_handler=str)
    print(f"\nSaved {len(out_df)} entries to {args.output}")
    print_distribution(
        "AFTER (actual)",
        out_df["duration"].to_numpy(),
        out_df["text"].str.len().to_numpy(),
    )


if __name__ == "__main__":
    main()
