import mlx_whisper

AUDIO = "ad.mp3"

result = mlx_whisper.transcribe(
    AUDIO,
    path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
    verbose=True,
)

with open("transcript.txt", "w") as f:
    f.write(result["text"])

with open("transcript_segments.txt", "w") as f:
    for seg in result["segments"]:
        f.write(f"[{seg['start']:.2f} -> {seg['end']:.2f}] {seg['text'].strip()}\n")

print(f"\nDone. {len(result['segments'])} segments written.")
