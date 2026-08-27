# WorkoutTrackerAI

Log a gym set by saying it out loud. Speak "three sets of eight at sixty kilos on bench",
and get back structured workout data instead of typing it into an app between sets.

A small voice pipeline: record → transcribe → structure.

```
microphone ──▶ PyAudio (16-bit, 44.1 kHz, mono)
           ──▶ voice_file.wav
           ──▶ Whisper (whisper-1)          transcription
           ──▶ GPT-4o-mini + system prompt   structured result
```

Both API calls are wrapped in a `@timer` decorator that prints wall-clock latency, so you
can see where the time actually goes — transcription usually dominates.

## Setup

```bash
pip install openai pyaudio python-dotenv
```

On Windows, `pyaudio` often fails to build from source. If it does, install a prebuilt
wheel instead: `pip install pipwin && pipwin install pyaudio`.

Then add your API key:

```bash
# .env
OPENAI_API_KEY=sk-...
```

## You also need a prompt.txt

`prompt.txt` is gitignored, so **the repo will not run as cloned** — `load_prompt()` raises
`ValueError: Incorrect file name` until you create one. It holds the system prompt that
turns a raw transcript into structured output. Something like:

```text
You are a gym workout parser. The user will describe a set they just completed,
in casual speech. Extract each exercise into structured form:

  exercise | sets | reps | weight | unit

Rules:
- If the user gives a range ("eight to ten reps"), take the upper bound.
- If no unit is stated, assume kilograms.
- If a field genuinely isn't stated, write "unknown" rather than guessing.
- Output only the rows. No commentary.
```

Tune it to the output you want — the pipeline is prompt-driven, so this file is where the
behaviour lives.

## Run it

```bash
python speech_recognition.py
```

Press `r` to record a 10-second clip, `q` to quit. Each recording overwrites
`voice_file.wav` in the working directory.

`speech_recognition_demo.ipynb` walks the same pipeline in stages — transcription first,
then the analysis layer on top — which is the easier place to start if you want to swap
models or see the intermediate transcript.

## Known limits

- Recording length is fixed at 10 seconds in `voice_to_text_with_analysis()`; there's no
  silence detection, so short sets still wait out the full clip.
- An audio-device failure degrades confusingly: `record_voice()` returns `None`, then
  `transcribe_text()` catches the resulting `TypeError` in its blanket `except Exception`
  and reports it as a *transcription* error. The run continues with `text=None` and only
  fails once the chat call rejects `None` as message content.
- Results are printed, not persisted. There's no store behind this yet.

## Licence

MIT.
