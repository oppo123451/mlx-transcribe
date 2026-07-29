import subprocess
import mlx_whisper
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

MODEL = "mlx-community/whisper-large-v3-mlx"
SR = 16000
vad = load_silero_vad()


def audio_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def is_hallucination(seg):
    text = seg["text"].strip()
    if not text:
        return True
    if seg.get("no_speech_prob", 0) > 0.6:
        return True
    if seg.get("avg_logprob", 0) < -1.0:
        return True
    if seg.get("compression_ratio", 0) > 2.4:
        return True
    words = text.lower().split()
    if len(words) > 4 and len(set(words)) / len(words) < 0.35:
        return True
    return False


def transcribe(path, context=None):
    duration = audio_duration(path)
    wav = read_audio(path, sampling_rate=SR)

    speech = get_speech_timestamps(
        wav, vad, sampling_rate=SR,
        min_speech_duration_ms=250,
        min_silence_duration_ms=300,
        speech_pad_ms=200,
    )

    segments = []
    for region in speech:
        offset = region["start"] / SR
        chunk = wav[region["start"]:region["end"]].numpy()

        result = mlx_whisper.transcribe(
            chunk,
            path_or_hf_repo=MODEL,
            temperature=0.0,
            condition_on_previous_text=False,
            no_speech_threshold=0.5,
            initial_prompt=context,
        )

        for seg in result["segments"]:
            if is_hallucination(seg):
                continue
            seg["start"] += offset
            seg["end"] += offset
            if seg["start"] >= duration:
                continue
            seg["end"] = min(seg["end"], duration)
            segments.append(seg)

    return {
        "duration": duration,
        "segments": segments,
        "text": " ".join(s["text"].strip() for s in segments),
    }


if __name__ == "__main__":
    out = transcribe("ad.mp3")
    for s in out["segments"]:
        print(f"[{s['start']:6.2f} -> {s['end']:6.2f}] {s['text'].strip()}")
