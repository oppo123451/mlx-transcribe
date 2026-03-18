# mlx-transcribe

hi im ryan!!

so whisper is very good at transcribing speech and very bad at
noticing when there isn't any. feed it silence and it will
confidently write you a sentence :)

my first test was a 30 second ad. it gave me a segment running
from 30.0s to 59.98s. the file is 30.1 seconds long!! it just
made up thirty seconds of audio that does not exist

so this gates whisper behind silero vad and throws away segments
that fail on log-probability, compression ratio, or lexical
repetition. runs entirely on your laptop. large-v3 on apple
silicon via mlx/metal, about 2.7x realtime

(it still gets proper nouns wrong. it wrote "tim davey" once and
"davy" four hundred seconds later, in the same transcript. turns
out turning off condition_on_previous_text to kill the
hallucination loops also kills spelling consistency across
segments. i think this is fine. i have not fixed it)

feel free to use this in your machine or pay openai $0.006/min :)

