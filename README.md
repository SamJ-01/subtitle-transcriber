# subtitle-transcriber

[![tests](https://github.com/SamJ-01/subtitle-transcriber/actions/workflows/tests.yml/badge.svg)](https://github.com/SamJ-01/subtitle-transcriber/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

A Python command-line tool that turns speech in an audio or video file into an
**SRT subtitle file**, in **any of the ~100 languages Whisper understands**,
with the option to translate into English.

It runs **entirely on your own computer** using
[faster-whisper](https://github.com/SYSTRAN/faster-whisper). Nothing is
uploaded anywhere.

## Contents

- [Why I built this](#why-i-built-this)
- [What it does](#what-it-does)
- [Installation](#installation)
- [Usage](#usage)
- [How it works](#how-it-works)
- [Tests](#tests)
- [Limitations](#limitations)
- [Responsible use](#responsible-use) · [Licence](#licence)

## Why I built this

I make YouTube reaction videos and needed accurate subtitles for clips in
other languages. The first version was a quick script with the file names
written into the code. This version is a reusable tool: options on the command
line, better subtitle timing, and automated tests.

The code was developed with help from AI tools (ChatGPT for the first
version, Claude for the review and rewrite). I've reviewed it and can explain
how every part works.

## What it does

1. **Takes an audio or video file**: mp3, mp4, m4a, wav, mkv and more.
2. **Detects the language automatically**, or uses the one you choose.
3. **Transcribes the speech** with Whisper, skipping silence and music
   (where speech models tend to invent text).
4. **Splits long sentences into short subtitles** using the time of each
   word, so every subtitle appears exactly when it's spoken.
5. **Writes a standard `.srt` file** that works in YouTube, VLC, Premiere Pro,
   DaVinci Resolve, CapCut and most other video tools.

## Installation

Requires **Python 3.9+**. You don't need to install FFmpeg separately:
faster-whisper includes what it needs to read audio and video.

```bash
git clone https://github.com/SamJ-01/subtitle-transcriber.git
cd subtitle-transcriber

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

The first time you run it, the chosen model is downloaded (the default
`medium` model is about 1.5 GB) and then reused.

## Usage

```bash
python transcribe.py audio.mp3                   # creates audio.srt
python transcribe.py movie.mp4 -o movie.srt      # choose the output name
python transcribe.py clip.mp3 --language ja      # tell it the language
python transcribe.py clip.mp3 --translate        # English subtitles from any language
python transcribe.py clip.mp3 --model small      # faster, slightly less accurate
python transcribe.py song.mp3 --no-vad           # songs and music videos
```

| Option | Meaning |
| --- | --- |
| `input` | The audio or video file to transcribe |
| `-o, --output` | Where to save the subtitles (default: the input's name, ending `.srt`) |
| `--language` | Language code, e.g. `en`, `es`, `fr`, `de`, `hi`, `ja`, `ko`, `ar`, `ru`, `pt` (default: detect automatically) |
| `--translate` | Translate the speech into English subtitles |
| `--no-vad` | Don't skip music. **Use this for songs**, otherwise most of the singing is skipped |
| `--model` | `tiny`, `base`, `small`, `medium` or `large-v3` (default: `medium`) |
| `--max-chars` | Longest a single subtitle can be, in characters (default: 42) |

While it runs, it shows its progress:

```text
Loading the 'medium' model (the first run downloads it)...
Transcribing clip.mp4...
Language: es (97% sure) | Length: 12.4 min

  2.1% | 3.2x real time | 0.1 min elapsed
  4.3% | 3.4x real time | 0.2 min elapsed
...
```

"3.2x real time" means a 30-minute video takes about 10 minutes.

### Choosing a model

| Model | Speed | Accuracy | Good for |
| --- | --- | --- | --- |
| `tiny` / `base` | Fastest | Lowest | Quick drafts in clear English |
| `small` | Fast | Good | Most English content |
| `medium` | Moderate | Very good | **Default**: other languages, accents, background noise |
| `large-v3` | Slowest | Best | Difficult audio, less common languages |

### Output format

Each subtitle has a number, a start and end time, and the text:

```text
1
00:00:00,000 --> 00:00:02,100
Hello everyone, welcome back to

2
00:00:02,100 --> 00:00:04,200
the channel.
```

## How it works

All the code is in [`transcribe.py`](transcribe.py), split into small
functions:

| Function | Job |
| --- | --- |
| `parse_args()` | Reads the options typed on the command line |
| `load_model()` | Loads the Whisper model in `int8` format, which is fast on a normal CPU |
| `split_into_subtitles()` | Cuts long sentences into subtitles of up to 42 characters, using each word's timing |
| `format_time()` | Turns seconds (`75.5`) into SRT time (`00:01:15,500`) |
| `write_srt()` | Writes each subtitle as soon as it's ready, and shows progress |
| `main()` | Puts it all together |

Key Whisper settings, and why:

| Setting | Why |
| --- | --- |
| `vad_filter=True` | Skips silence and music, the main cause of made-up lines like "Thanks for watching!" |
| `word_timestamps=True` | Times every word, so subtitles can be split accurately |
| `condition_on_previous_text=False` | Stops it getting stuck repeating the same line in long videos |
| `beam_size=1`, `temperature=0` | Fastest decoding; always picks the most likely words |

## Tests

```bash
pip install -r requirements-dev.txt
pytest -v
```

The tests **never load the real model**. They use small fake words and
segments shaped exactly like faster-whisper's output, so they run in under a
second and need no download. GitHub Actions runs them on every push (see the
badge at the top).

They check:

- **Any language works**: Spanish, Hindi, Japanese (which has no spaces),
  Arabic and Russian text all come through correctly.
- **Options**: automatic language detection by default; `--language`,
  `--translate` and `--no-vad` are passed to Whisper correctly.
- **Subtitle splitting**: long sentences are split at the right words, with
  the right times.
- **Time formatting**: including rounding, e.g. 2.5 s → `00:00:02,500`, not
  `00:00:02,499`.
- **A complete SRT file** compared line by line with the expected output.
- **Edge cases**: empty segments, missing word timings, zero-length audio and
  a missing input file.

Example run:

```text
transcribe.py::transcribe.format_time PASSED
tests/test_transcribe.py::test_format_time[0-00:00:00,000] PASSED
tests/test_transcribe.py::test_format_time[2.5-00:00:02,500] PASSED
tests/test_transcribe.py::test_format_time[75.5-00:01:15,500] PASSED
tests/test_transcribe.py::test_format_time[59.9996-00:01:00,000] PASSED
tests/test_transcribe.py::test_format_time[3600-01:00:00,000] PASSED
tests/test_transcribe.py::test_format_time[5025.125-01:23:45,125] PASSED
tests/test_transcribe.py::test_short_segment_stays_as_one_subtitle PASSED
tests/test_transcribe.py::test_long_segment_is_split_using_word_times PASSED
tests/test_transcribe.py::test_single_word_longer_than_limit_is_kept_whole PASSED
tests/test_transcribe.py::test_any_language_is_written_correctly[spanish] PASSED
tests/test_transcribe.py::test_any_language_is_written_correctly[hindi] PASSED
tests/test_transcribe.py::test_any_language_is_written_correctly[japanese] PASSED
tests/test_transcribe.py::test_any_language_is_written_correctly[arabic] PASSED
tests/test_transcribe.py::test_any_language_is_written_correctly[russian] PASSED
tests/test_transcribe.py::test_language_is_detected_automatically_by_default PASSED
tests/test_transcribe.py::test_language_can_be_chosen PASSED
tests/test_transcribe.py::test_translate_option_asks_for_english PASSED
tests/test_transcribe.py::test_music_filter_is_on_by_default PASSED
tests/test_transcribe.py::test_no_vad_option_keeps_music PASSED
tests/test_transcribe.py::test_full_srt_output PASSED
tests/test_transcribe.py::test_segment_without_word_times_uses_segment_times PASSED
tests/test_transcribe.py::test_zero_length_audio_does_not_crash PASSED
tests/test_transcribe.py::test_output_name_can_be_chosen PASSED
tests/test_transcribe.py::test_missing_file_is_a_clear_error PASSED

25 passed in 0.06s
```

## Limitations

- **Accuracy varies by language.** Whisper is strongest in widely spoken
  languages; for less common ones, use `--model large-v3` and check the result.
- **Songs need `--no-vad`.** The silence/music filter is tuned for speech,
  so it treats singing over a backing track as music and skips it. In a test
  on a 4.7-minute pop song with the filter on, it wrote only 11 subtitles.
- **Always proofread.** Names, slang, songs and overlapping speech can be
  misheard.
- `--translate` only translates **into English** (a Whisper limitation).
- `--max-chars` counts characters, so for languages like Japanese or Chinese,
  where each character carries more meaning, a lower value such as 20 may
  read better.
- It runs on the CPU. Long films can take a while on older machines; use a
  smaller model for speed.

## Responsible use

Transcribing a video for your own editing is fine, but the words in someone
else's film or video are still their copyright. Check the platform's rules and
fair-use / fair-dealing guidance before publishing subtitles of other people's
content.

## Licence

Released under the [MIT Licence](LICENSE).
