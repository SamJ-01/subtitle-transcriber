"""Transcribe an audio or video file into an SRT subtitle file.

Uses faster-whisper (an optimised version of OpenAI's Whisper speech
recognition model), which understands around 100 languages and runs entirely
on your own computer. Nothing is uploaded anywhere.

Examples:
    python transcribe.py audio.mp3                   # detect the language
    python transcribe.py movie.mp4 -o movie.srt      # choose the output name
    python transcribe.py clip.mp3 --language ja      # tell it the language
    python transcribe.py clip.mp3 --translate        # English subtitles
    python transcribe.py song.mp3 --no-vad           # songs and music videos
"""

import argparse
import sys
import time
from pathlib import Path

# Print a progress line every time we get this many percent further through.
PROGRESS_STEP = 2


def parse_args(argv=None):
    """Read the options the user typed on the command line."""
    parser = argparse.ArgumentParser(
        description="Transcribe an audio or video file into an SRT subtitle file."
    )
    parser.add_argument("input", help="the audio or video file to transcribe")
    parser.add_argument(
        "-o", "--output",
        help="where to save the subtitles (default: same name as the input, ending .srt)",
    )
    parser.add_argument(
        "--model", default="medium",
        help="Whisper model size: tiny, base, small, medium or large-v3. "
             "Bigger is more accurate but slower (default: medium)",
    )
    parser.add_argument(
        "--language",
        help="language code such as en, es, fr, hi, ja or ar "
             "(default: detect it automatically)",
    )
    parser.add_argument(
        "--translate", action="store_true",
        help="translate the speech into English subtitles instead of "
             "keeping the original language",
    )
    parser.add_argument(
        "--no-vad", action="store_true",
        help="don't skip parts that sound like music or silence. Use this "
             "for songs, where the singing would otherwise be skipped",
    )
    parser.add_argument(
        "--max-chars", type=int, default=42,
        help="longest a single subtitle can be, in characters (default: 42)",
    )
    # argv=None means "use what was typed on the command line". The tests
    # pass in their own list instead.
    return parser.parse_args(argv)


def format_time(seconds):
    """Turn a number of seconds into SRT time format.

    >>> format_time(75.5)
    '00:01:15,500'
    """
    # Work in whole milliseconds, rounding rather than cutting off, so that
    # 2.5 seconds becomes 2,500 and not 2,499.
    total_ms = round(seconds * 1000)
    # divmod(a, b) gives two answers at once: how many whole b's fit in a,
    # and what is left over.
    hours, total_ms = divmod(total_ms, 3_600_000)   # 3,600,000 ms in an hour
    minutes, total_ms = divmod(total_ms, 60_000)    # 60,000 ms in a minute
    secs, millis = divmod(total_ms, 1000)           # 1,000 ms in a second
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def make_subtitle(words):
    """Turn a list of words into one subtitle: (start time, end time, text)."""
    # Whisper includes any space a word needs at the start of the word itself
    # (" Hello"), and languages like Japanese and Chinese have no spaces at
    # all, so we join the words with nothing in between.
    text = "".join(word.word for word in words).strip()
    return (words[0].start, words[-1].end, text)


def split_into_subtitles(words, max_chars):
    """Split one segment's words into subtitles no longer than max_chars.

    Whisper's segments can be long sentences. Using the timing of each word,
    we cut them into shorter subtitles that appear exactly when they are spoken.
    """
    subtitles = []
    current = []  # the words in the subtitle we are building

    for word in words:
        # How long would the subtitle be if we added this word?
        new_text = "".join(w.word for w in current + [word]).strip()
        if current and len(new_text) > max_chars:
            # Too long: finish the current subtitle and start a new one.
            subtitles.append(make_subtitle(current))
            current = []
        current.append(word)

    # Don't forget the last subtitle.
    if current:
        subtitles.append(make_subtitle(current))
    return subtitles


def show_progress(position, duration, start_time):
    """Print how far through the file we are and how fast it's going."""
    elapsed = time.time() - start_time
    percent = position / duration * 100
    # "Speed" is seconds of audio processed per second of real time,
    # so 3.0x means a 30-minute video takes about 10 minutes.
    speed = position / elapsed if elapsed > 0 else 0
    print(f"{percent:5.1f}% | {speed:.1f}x real time | {elapsed / 60:.1f} min elapsed")


def write_srt(segments, duration, output_path, max_chars):
    """Write Whisper's segments to an SRT file. Returns how many subtitles."""
    start_time = time.time()
    count = 0          # how many subtitles we've written so far
    last_reported = 0  # the percentage we last printed progress at

    with open(output_path, "w", encoding="utf-8") as srt_file:
        # Segments are produced one at a time as Whisper works through the
        # audio, so subtitles are saved as we go rather than all at the end.
        for segment in segments:
            if segment.words:
                subtitles = split_into_subtitles(segment.words, max_chars)
            else:
                subtitles = [(segment.start, segment.end, segment.text.strip())]

            for start, end, text in subtitles:
                if not text:
                    continue  # skip empty subtitles
                count += 1
                srt_file.write(f"{count}\n")
                srt_file.write(f"{format_time(start)} --> {format_time(end)}\n")
                srt_file.write(f"{text}\n\n")

            # Only show progress if we know the length (avoids dividing by zero).
            if duration > 0:
                percent = segment.end / duration * 100
                if percent - last_reported >= PROGRESS_STEP:
                    show_progress(segment.end, duration, start_time)
                    last_reported = percent
    return count


def load_model(model_size):
    """Load the Whisper model (the first run downloads it)."""
    # Imported here rather than at the top of the file, so the tests can run
    # without faster-whisper (a large download) being installed.
    from faster_whisper import WhisperModel

    # int8 stores the model's numbers in a smaller format: much faster on a
    # normal CPU, with very little loss of accuracy.
    return WhisperModel(model_size, compute_type="int8")


def main(argv=None):
    args = parse_args(argv)

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        return 1

    # If no output name was given, use the input's name with .srt on the end.
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix(".srt")

    print(f"Loading the '{args.model}' model (the first run downloads it)...")
    model = load_model(args.model)

    print(f"Transcribing {input_path.name}...")
    start_time = time.time()

    segments, info = model.transcribe(
        str(input_path),
        # "translate" turns any language into English; "transcribe" keeps it.
        task="translate" if args.translate else "transcribe",
        language=args.language,     # None means "detect it automatically"
        beam_size=1,                # check one guess at a time: fastest option
        temperature=0,              # always pick the most likely words
        condition_on_previous_text=False,  # stops it getting stuck repeating a line
        # Skip silence and music, where Whisper invents text. --no-vad turns
        # this off for songs, because sung vocals don't sound like speech.
        vad_filter=not args.no_vad,
        word_timestamps=True,       # time every word, so we can split subtitles accurately
    )
    print(
        f"Language: {info.language} ({info.language_probability:.0%} sure) | "
        f"Length: {info.duration / 60:.1f} min\n"
    )

    count = write_srt(segments, info.duration, output_path, args.max_chars)

    minutes_taken = (time.time() - start_time) / 60
    print(f"\nDone: wrote {count} subtitles to {output_path} in {minutes_taken:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
