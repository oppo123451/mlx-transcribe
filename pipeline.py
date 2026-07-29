"""
Local audio transcription with VAD preprocessing and hallucination filtering.

Decodes via FFmpeg (any format), segments speech with Silero VAD, transcribes
each region with Whisper large-v3 on MLX, then filters low-confidence output.
"""

import subprocess

import mlx_whisper
import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

MODEL = "mlx-community/whisper-large-v3-mlx"
SR = 16000

_vad = None


def get_vad():
    """Load the VAD model once and reuse it."""
    global _vad
    if _vad is None:
        _vad = load_silero_vad()
    return _vad


def load_audio(path, sr=SR):
    """
    Decode any FFmpeg-supported format to mono float32 at the target rate.

    Avoids torchaudio, whose format support has narrowed across recent
    releases and is unreliable for mp3.
    """
    cmd = [
        "ffmpeg", "-nostdin", "-threads", "0",
        "-i", str(path),
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"FFmpeg failed to decode {path}:\n{e.stderr.decode(errors='replace')}"
        ) from e

    audio = np.frombuffer(out, np.int16).astype(np.float32) / 32768.0
    if audio.size == 0:
        raise RuntimeError(f"No audio decoded from {path} — corrupt or empty file?")
    return audio


def is_hallucination(seg):
    """
    Reject segments on Whisper's own confidence signals.

    None of these thresholds are file-specific. They are starting values —
    tighten to drop more false text, loosen to keep more quiet real speech.
    """
    text = seg["text"].strip()
    if not text:
        return True
    if seg.get("no_speech_prob", 0.0) > 0.6:
        return True
    if seg.get("avg_logprob", 0.0) < -1.0:
        return True
    # High compression ratio means highly repetitive text — a loop signature.
    if seg.get("compression_ratio", 0.0) > 2.4:
        return True
    words = text.lower().split()
    if len(words) > 4 and len(set(words)) / len(words) < 0.35:
        return True
    return False


def transcribe(path, context=None, model=MODEL):
    """
    Transcribe an audio file.

    Args:
        path:    Path to any FFmpeg-decodable audio or video file.
        context: Optional domain vocabulary to bias decoding — names,
                 acronyms, technical terms. Improves recognition of words
                 outside the model's typical distribution.
        model:   Hugging Face repo for the MLX Whisper weights.

    Returns:
        dict with 'duration', 'segments', and 'text'.
    """
    audio = load_audio(path)
    duration = len(audio) / SR

    regions = get_speech_timestamps(
        torch.from_numpy(audio),
        get_vad(),
        sampling_rate=SR,
        min_speech_duration_ms=250,
        min_silence_duration_ms=300,
        speech_pad_ms=200,
    )

    segments = []
    for region in regions:
        offset = region["start"] / SR
        chunk = audio[region["start"]:region["end"]]

        result = mlx_whisper.transcribe(
            chunk,
            path_or_hf_repo=model,
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
            # Whisper pads the final 30s window and can emit segments past
            # the true end of the audio. Nothing real starts after EOF.
            if seg["start"] >= duration:
                continue
            seg["end"] = min(seg["end"], duration)
            segments.append(seg)

    segments.sort(key=lambda s: s["start"])

    return {
        "duration": duration,
        "segments": segments,
        "text": " ".join(s["text"].strip() for s in segments),
    }


def to_srt(segments):
    """Export segments as SRT subtitles."""

    def stamp(t):
        h, rem = divmod(t, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int((s % 1) * 1000):03d}"

    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{stamp(seg['start'])} --> {stamp(seg['end'])}")
        lines.append(seg["text"].strip())
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "ad.mp3"
    out = transcribe(target)

    print(f"\n{target} — {out['duration']:.1f}s, {len(out['segments'])} segments\n")
    for s in out["segments"]:
        print(f"[{s['start']:7.2f} -> {s['end']:7.2f}] {s['text'].strip()}")

    with open("transcript.txt", "w") as f:
        f.write(out["text"])
    with open("transcript.srt", "w") as f:
        f.write(to_srt(out["segments"]))
