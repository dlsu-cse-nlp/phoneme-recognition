import re

import pandas as pd
import panphon
import panphon.distance

ACTUAL = "na'tapon ʔaŋ t͡ʃa'ʔa sa 'd͡ʒaket"
INPUT = "ʔaŋ d͡ʒun'joɾ 'haj 'skul"


class Metrics:
    def __init__(self):
        self.pft = panphon.FeatureTable()
        self.sft = panphon.FeatureTable()
        self.dst = panphon.distance.Distance()

        self.reset()

        raw_strings = [
            seg[0] if isinstance(seg, tuple) else seg for seg in self.sft.segments
        ]
        existing_segs = sorted(raw_strings, key=len, reverse=True)
        escaped_segs = [re.escape(seg) for seg in existing_segs]

        patterns = [r"\s+", r"\'"]
        re_string = "|".join(patterns + escaped_segs)
        compiled_re = re.compile(re_string)

        self.sft.ipa_segs = lambda text: compiled_re.findall(text)

    def reset(self):
        """Resets the state for output/pooledmetrics"""

        self.output = []

        self.total_per_dist = 0
        self.total_pfer_dist = 0
        self.total_sper_dist = 0

        self.total_per_segs = 0
        self.total_pfer_segs = 0
        self.total_sper_segs = 0

        self.total_per_segs

    def get_sample_per(self, pred, actual, save=False):
        """Calculates PER for a single sample, and optionally saves to pool"""

        pred_segs = self.pft.ipa_segs(pred)
        actual_segs = self.pft.ipa_segs(actual)
        seg_len = len(actual_segs) if actual_segs else np.nan

        per_dist = self.dst.levenshtein_distance(pred_segs, actual_segs)

        if save:
            self.total_per_dist += per_dist
            self.total_per_segs += seg_len

        return per_dist / seg_len

    def get_sample_pfer(self, pred, actual, save=False):
        """Calculates PFER for a single sample, and optionally saves to pool"""

        # NOTE: This uses the default PanPhon feature table.
        actual_segs = self.dst.fm.ipa_segs(actual)
        pfer_dist = self.dst.feature_edit_distance(pred, actual)
        seg_len = len(actual_segs) if actual_segs else np.nan

        if save:
            self.total_pfer_dist += pfer_dist
            self.total_pfer_segs += seg_len

        return pfer_dist / seg_len

    def get_sample_sper(self, pred, actual, save=False):
        """Calculates SPER for a single sample, and optionally saves to pool"""

        pred_segs = self.sft.ipa_segs(pred)
        actual_segs = self.sft.ipa_segs(actual)
        seg_len = len(actual_segs) if actual_segs else np.nan

        sper_dist = self.dst.levenshtein_distance(pred_segs, actual_segs)

        if save:
            self.total_sper_dist += sper_dist
            self.total_sper_segs += seg_len

        return sper_dist / seg_len

    def get_sample_all(self, pred, actual, save=False, sentence=None, verbose=True):
        """Calculates all three metrics for a single sample"""

        metrics = {
            "target": actual,
            "predicted": pred,
            "per": self.get_sample_per(pred, actual, save),
            "pfer": self.get_sample_pfer(pred, actual, save),
            "sper": self.get_sample_sper(pred, actual, save),
        }

        if verbose:
            print("-" * 40)
            print(f"Target: {metrics['target']}")
            print(f"Predict: {metrics['predicted']}")
            print(f"PER: {metrics['per']}")
            print(f"PFER: {metrics['pfer']}")
            print(f"SPER: {metrics['sper']}")

        if save:
            self.output.append(metrics)

        return metrics

    def get_pooled_metrics(self):
        # TODO: Maybe also directly report error count/segment lengths?
        return {
            "PER": self.total_per_dist / self.total_per_segs,
            "PFER": self.total_pfer_dist / self.total_pfer_segs,
            "SPER": self.total_sper_dist / self.total_sper_segs,
        }

    def get_output_df(self):
        return pd.DataFrame(self.output)


metrics = Metrics()
print(metrics.pft.ipa_segs(ACTUAL))
print(metrics.sft.ipa_segs(ACTUAL))
