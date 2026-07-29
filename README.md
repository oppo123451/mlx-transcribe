# mlx-transcribe

fast, private speech-to-text that runs entirely on your own mac :)

no api keys. no cloud. no per-minute billing. your audio never leaves the
machine. whisper large-v3 runs on the apple silicon gpu through mlx, at
roughly **2.7x realtime** — an eight minute recording transcribes in about
three minutes.

## what it does

- **transcribes audio and video** — mp3, wav, m4a, flac, aiff, mp4, mov,
  anything ffmpeg can decode
- **timestamps every segment** — start and end times to the centisecond
- **exports subtitles** — plain text and `.srt` out of the box
- **detects speech first** — silero vad finds the parts where someone is
  actually talking, so music, silence, and room tone never reach the model
- **filters low-confidence output** — segments are scored on log-probability,
  compression ratio, and lexical diversity, and dropped if they fail
- **takes domain context** — pass in names, acronyms, or jargon and the
  decoder biases toward them. big accuracy win on proper nouns
- **runs offline** — after the first model download, no network needed at all

## requirements

| | |
|---|---|
| machine | apple silicon mac (m1 or later) |
| python | 3.10+ |
| ffmpeg | `brew install ffmpeg` |
| disk | ~3 gb for model weights |

## install

```bash
git clone https://github.com/oppo123451/mlx-transcribe.git
cd mlx-transcribe
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

model weights download automatically the first time you run it and cache to
`~/.cache/huggingface`, so it only happens once.

## usage

command line:

```bash
python pipeline.py interview.mp3
```

prints timestamped segments and writes `transcript.txt` and `transcript.srt`.

as a library:

```python
from pipeline import transcribe

result = transcribe("interview.mp3")

print(result["text"])          # full transcript
print(result["duration"])      # length in seconds

for seg in result["segments"]:
    print(seg["start"], seg["end"], seg["text"])
```

with domain context — this is the single biggest accuracy lever, use it:

```python
result = transcribe(
    "earnings_call.mp3",
    context="EBITDA, basis points, year-over-year, NASDAQ, Q3 guidance",
)
```

pick a different model size to trade accuracy for speed:

```python
result = transcribe("podcast.mp3", model="mlx-community/whisper-small-mlx")
```

## how it works

```
audio file
    |  ffmpeg                        decode any codec -> 16 khz mono float32
    |  silero vad                    find speech regions, skip everything else
    |  whisper large-v3 (mlx/metal)  transcribe each region
    |  confidence filter             drop low-scoring segments
    |  timestamp offset + clamp      map back to the real timeline
transcript + srt
```

**the stack**

- [**mlx**](https://github.com/ml-explore/mlx) — apple's array framework.
  runs the model on the mac gpu through metal, using unified memory so there
  is no cpu/gpu copying
- [**whisper large-v3**](https://github.com/openai/whisper) — openai's
  encoder-decoder asr model, 99 languages, the full size version
- [**silero vad**](https://github.com/snakers4/silero-vad) — tiny, fast
  voice activity detection model
- [**ffmpeg**](https://ffmpeg.org/) — decodes literally any audio or video
  format you throw at it

## license

mit — do whatever you want with it :)
