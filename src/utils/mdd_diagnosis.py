import difflib


def align_phonemes(canonical, recognized, delimiter=" "):
    # Aligns the canonical phonemes against recognized (actually produced) phonemes
    # returns a list of diagnostic events
    if delimiter:
        canon_tokens = canonical.strip().split(delimiter)
        rec_tokens = recognized.strip().split(delimiter)
    else:
        canon_tokens = list(canonical.strip())
        rec_tokens = list(recognized.strip())

    matcher = difflib.SequenceMatcher(a=canon_tokens, b=rec_tokens, autojunk=False)
    events = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                events.append(
                    {
                        "type": "correct",
                        "expected": canon_tokens[i1 + k],
                        "actual": rec_tokens[j1 + k],
                    }
                )
        elif tag == "replace":
            # Pair substitutions position-by-position where both sides still
            # have tokens left; anything left over is a pure insertion or
            # deletion.
            span = max(i2 - i1, j2 - j1)
            for k in range(span):
                exp = canon_tokens[i1 + k] if i1 + k < i2 else None
                act = rec_tokens[j1 + k] if j1 + k < j2 else None
                if exp is not None and act is not None:
                    events.append(
                        {"type": "substitution", "expected": exp, "actual": act}
                    )
                elif exp is not None:
                    events.append({"type": "deletion", "expected": exp, "actual": None})
                else:
                    events.append(
                        {"type": "insertion", "expected": None, "actual": act}
                    )
        elif tag == "delete":
            for k in range(i1, i2):
                events.append(
                    {"type": "deletion", "expected": canon_tokens[k], "actual": None}
                )
        elif tag == "insert":
            for k in range(j1, j2):
                events.append(
                    {"type": "insertion", "expected": None, "actual": rec_tokens[k]}
                )

    return events


def unit_index_per_byte(ids, delimiter_id):
    """Given a list of byte-level token ids and the id of the delimiter byte
    (e.g. the id for " "), returns a parallel list giving, for each byte
    position, the index of the phoneme unit (as produced by
    `text.split(delimiter)`) that byte belongs to.
    """
    if delimiter_id is None:
        return list(range(len(ids)))

    unit_of_byte = []
    unit_idx = 0
    for tok in ids:
        unit_of_byte.append(unit_idx)
        if tok == delimiter_id:
            unit_idx += 1
    return unit_of_byte
