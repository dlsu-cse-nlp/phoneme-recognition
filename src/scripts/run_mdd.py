from pathlib import Path

import torch
import torchaudio

from src.engines.mdd_build import load_mdd_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "models/checkpoints/mdd_model_initialized"


def load_audio(path, target_sr=16000):
    waveform, sr = torchaudio.load(path)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # downmix to mono
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform.squeeze(0).numpy()


def write_textgrid(output_path, result, sampling_rate=16000):
    # Writes Praat TextGrid file with frame-by-frame mispronunciation diagnosis
    total_duration = result["total_duration"]
    frame_duration = result["frame_duration"]
    frame_reports = result["frame_reports"]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        # File header
        f.write('File type = "ooTextFile"\n')
        f.write('Object class = "TextGrid"\n\n')
        f.write("xmin = 0\n")
        f.write(f"xmax = {total_duration:.6f}\n")
        f.write("tiers? <exists>\n")
        f.write("size = 1\n")  # One tier for frame diagnoses
        f.write("item []:\n")  # NOTE: needs the trailing colon, Praat requires it
        f.write("    item [1]:\n")
        f.write('        class = "IntervalTier"\n')
        f.write('        name = "MDD_Frames"\n')
        f.write("        xmin = 0\n")
        f.write(f"        xmax = {total_duration:.6f}\n")
        f.write(f"        intervals: size = {len(frame_reports)}\n")

        # Write out every interval segment sequentially
        for i, fr in enumerate(frame_reports):
            # Compute timing limits
            t_start = fr["frame"] * frame_duration
            t_end = min(t_start + frame_duration, total_duration)

            # Ensure the final frame fills the remaining file boundaries precisely
            if i == len(frame_reports) - 1:
                t_end = total_duration

            # Build label: e.g., "p->b (MIS)" or "p (ok)"
            if fr["expected"] is not None and fr["actual"] is not None:
                if fr["mispronounced"]:
                    label = f'{fr["expected"]}->{fr["actual"]} (MIS)'
                else:
                    label = f'{fr["expected"]} (ok)'
            else:
                label = "sil" if not fr["mispronounced"] else "MIS"

            f.write(f"        intervals [{i + 1}]:\n")
            f.write(f"            xmin = {t_start:.6f}\n")
            f.write(f"            xmax = {t_end:.6f}\n")
            f.write(f'            text = "{label}"\n')

    print(f"Successfully saved Praat alignment timeline: {output_path}")


def diagnose(
    model, tokenizer, feature_extractor, audio_path, expected_text, delimiter=" "
):
    waveform = load_audio(audio_path, target_sr=feature_extractor.sampling_rate)
    audio_inputs = feature_extractor(
        waveform,
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )
    audio_values = audio_inputs["input_values"].to(DEVICE)
    audio_attention_mask = audio_inputs.get("attention_mask")
    if audio_attention_mask is not None:
        audio_attention_mask = audio_attention_mask.to(DEVICE)

    return model.analyze(
        audio_values,
        tokenizer,
        expected_text,
        audio_attention_mask=audio_attention_mask,
        delimiter=delimiter,
        sampling_rate=feature_extractor.sampling_rate,
    )


if __name__ == "__main__":
    # Loaded ONCE -- everything below reuses model/tokenizer/feature_extractor
    model, tokenizer, feature_extractor = load_mdd_model(CHECKPOINT_DIR, device=DEVICE)

    result = diagnose(
        model,
        tokenizer,
        feature_extractor,
        audio_path="data/nexdata/filipino_822/G00001/G00001S0978.wav",
        expected_text="Ang unang araw ng Hulyo ay ang pagsisimula ng klase sa kolehiyo.",
    )
    print(f"Expected pronunciation   : {result['expected_pronunciation']}")
    print(f"Recognized pronunciation : {result['recognized_text']}")
    write_textgrid(
        "data/textgrids/my_custom_audio.TextGrid",
        result,
        sampling_rate=feature_extractor.sampling_rate,
    )
