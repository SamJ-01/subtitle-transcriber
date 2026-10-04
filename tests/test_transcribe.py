"""Tests for transcribe.py.

These never load the real Whisper model. Instead they use small, hand-made
"fake" words and segments shaped exactly like the ones faster-whisper
returns, so the tests run in well under a second and need no download.
"""

from types import SimpleNamespace

import pytest

import transcribe


# --- helpers that build fake Whisper output -------------------------------- #
def word(text, start, end):
    """A fake word, shaped like faster-whisper's Word objects."""
    return SimpleNamespace(word=text, start=start, end=end)


def segment(words, text=None):
    """A fake segment made from a list of fake words."""
    if text is None:
        text = "".join(w.word for w in words)
    return SimpleNamespace(
        start=words[0].start if words else 0.0,
        end=words[-1].end if words else 0.0,
        text=text,
        words=words,
    )


class FakeModel:
    """Stands in for WhisperModel and remembers how it was called."""

    def __init__(self, segments, language="en", duration=10.0, probability=0.98):
        self.segments = segments
        self.info = SimpleNamespace(
            language=language, language_probability=probability, duration=duration
        )
        self.called_with = None

    def transcribe(self, path, **options):
        self.called_with = options
        return iter(self.segments), self.info


def run(tmp_path, monkeypatch, segments, extra_args=(), **model_info):
    """Run the whole program against a fake model and return the SRT text."""
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"not really audio")
    model = FakeModel(segments, **model_info)
    monkeypatch.setattr(transcribe, "load_model", lambda size: model)

    exit_code = transcribe.main([str(audio), *extra_args])

    assert exit_code == 0
    srt = (tmp_path / "clip.srt").read_text(encoding="utf-8")
    return srt, model


# --- format_time ----------------------------------------------------------- #
@pytest.mark.parametrize(
    "seconds, expected",
    [
        (0, "00:00:00,000"),
        (2.5, "00:00:02,500"),          # rounded, not cut off to 2,499
        (75.5, "00:01:15,500"),
        (59.9996, "00:01:00,000"),      # rounds up into the next minute
        (3600, "01:00:00,000"),
        (5025.125, "01:23:45,125"),
    ],
)
def test_format_time(seconds, expected):
    assert transcribe.format_time(seconds) == expected


# --- split_into_subtitles -------------------------------------------------- #
def test_short_segment_stays_as_one_subtitle():
    words = [word(" Hello", 0.0, 0.5), word(" world.", 0.5, 1.0)]
    assert transcribe.split_into_subtitles(words, 42) == [(0.0, 1.0, "Hello world.")]


def test_long_segment_is_split_using_word_times():
    words = [
        word(" This", 0.0, 0.3), word(" is", 0.3, 0.5), word(" a", 0.5, 0.6),
        word(" long", 0.6, 1.0), word(" sentence.", 1.0, 1.8),
    ]
    subtitles = transcribe.split_into_subtitles(words, 10)
    assert subtitles == [
        (0.0, 0.6, "This is a"),
        (0.6, 1.0, "long"),
        (1.0, 1.8, "sentence."),
    ]
    # Every subtitle fits the limit.
    assert all(len(text) <= 10 for _, _, text in subtitles)


def test_single_word_longer_than_limit_is_kept_whole():
    words = [word(" Supercalifragilistic", 0.0, 2.0)]
    assert transcribe.split_into_subtitles(words, 5) == [
        (0.0, 2.0, "Supercalifragilistic")
    ]


# --- different languages --------------------------------------------------- #
# Whisper returns words in the same shape whatever the language. These check
# that spacing, accents and non-Latin scripts all survive into the SRT file.
LANGUAGE_SAMPLES = {
    "spanish": ([" ¿Cómo", " estás?"], "¿Cómo estás?"),
    "hindi": ([" नमस्ते", " दोस्तों"], "नमस्ते दोस्तों"),
    "japanese": (["こんにちは", "世界"], "こんにちは世界"),   # no spaces
    "arabic": ([" مرحبا", " بالعالم"], "مرحبا بالعالم"),
    "russian": ([" Привет,", " мир!"], "Привет, мир!"),
}


@pytest.mark.parametrize("language", LANGUAGE_SAMPLES)
def test_any_language_is_written_correctly(tmp_path, monkeypatch, language):
    pieces, expected = LANGUAGE_SAMPLES[language]
    words = [word(text, i, i + 1.0) for i, text in enumerate(pieces)]

    srt, _ = run(tmp_path, monkeypatch, [segment(words)])

    assert expected in srt


def test_language_is_detected_automatically_by_default(tmp_path, monkeypatch):
    _, model = run(tmp_path, monkeypatch, [segment([word(" Hi", 0, 1)])])
    assert model.called_with["language"] is None
    assert model.called_with["task"] == "transcribe"


def test_language_can_be_chosen(tmp_path, monkeypatch):
    _, model = run(tmp_path, monkeypatch, [segment([word(" Hola", 0, 1)])],
                   extra_args=["--language", "es"])
    assert model.called_with["language"] == "es"


def test_unsure_language_guess_gives_a_warning(tmp_path, monkeypatch, capsys):
    run(tmp_path, monkeypatch, [segment([word(" Hi", 0, 1)])], probability=0.24)
    assert "not sure about the language" in capsys.readouterr().out


def test_confident_language_guess_gives_no_warning(tmp_path, monkeypatch, capsys):
    run(tmp_path, monkeypatch, [segment([word(" Hi", 0, 1)])], probability=0.9)
    assert "not sure" not in capsys.readouterr().out


def test_no_warning_when_language_is_chosen(tmp_path, monkeypatch, capsys):
    run(tmp_path, monkeypatch, [segment([word(" Hola", 0, 1)])],
        extra_args=["--language", "es"], probability=0.24)
    assert "not sure" not in capsys.readouterr().out


def test_multilingual_is_off_by_default(tmp_path, monkeypatch):
    _, model = run(tmp_path, monkeypatch, [segment([word(" Hi", 0, 1)])])
    assert model.called_with["multilingual"] is False


def test_multilingual_option(tmp_path, monkeypatch):
    _, model = run(tmp_path, monkeypatch, [segment([word(" Hi", 0, 1)])],
                   extra_args=["--multilingual"])
    assert model.called_with["multilingual"] is True


def test_loop_protection_is_not_disabled(tmp_path, monkeypatch):
    # Forcing temperature=0 turns off faster-whisper's retry when a section
    # gets stuck repeating itself, so we must leave it at the default.
    _, model = run(tmp_path, monkeypatch, [segment([word(" Hi", 0, 1)])])
    assert "temperature" not in model.called_with


def test_translate_option_asks_for_english(tmp_path, monkeypatch):
    _, model = run(tmp_path, monkeypatch, [segment([word(" Hello", 0, 1)])],
                   extra_args=["--translate"])
    assert model.called_with["task"] == "translate"


def test_music_filter_is_on_by_default(tmp_path, monkeypatch):
    _, model = run(tmp_path, monkeypatch, [segment([word(" Hi", 0, 1)])])
    assert model.called_with["vad_filter"] is True


def test_no_vad_option_keeps_music(tmp_path, monkeypatch):
    _, model = run(tmp_path, monkeypatch, [segment([word(" Hi", 0, 1)])],
                   extra_args=["--no-vad"])
    assert model.called_with["vad_filter"] is False


# --- the finished SRT file ------------------------------------------------- #
def test_full_srt_output(tmp_path, monkeypatch):
    segments = [
        segment([word(" Hello", 0.0, 0.5), word(" there.", 0.5, 1.2)]),
        segment([], text="   "),                       # empty: skipped
        segment([word(" Goodbye.", 3725.0, 3727.25)]),
    ]
    srt, _ = run(tmp_path, monkeypatch, segments)

    assert srt == (
        "1\n"
        "00:00:00,000 --> 00:00:01,200\n"
        "Hello there.\n"
        "\n"
        "2\n"
        "01:02:05,000 --> 01:02:07,250\n"
        "Goodbye.\n"
        "\n"
    )


def test_segment_without_word_times_uses_segment_times(tmp_path, monkeypatch):
    seg = SimpleNamespace(start=1.0, end=2.0, text=" Fallback.", words=None)
    srt, _ = run(tmp_path, monkeypatch, [seg])
    assert "00:00:01,000 --> 00:00:02,000\nFallback." in srt


def test_zero_length_audio_does_not_crash(tmp_path, monkeypatch):
    srt, _ = run(tmp_path, monkeypatch, [segment([word(" Hi", 0, 0)])], duration=0)
    assert "Hi" in srt


def test_output_name_can_be_chosen(tmp_path, monkeypatch):
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"")
    model = FakeModel([segment([word(" Hi", 0, 1)])])
    monkeypatch.setattr(transcribe, "load_model", lambda size: model)

    assert transcribe.main([str(audio), "-o", str(tmp_path / "custom.srt")]) == 0
    assert (tmp_path / "custom.srt").exists()


def test_missing_file_is_a_clear_error(tmp_path, capsys):
    assert transcribe.main([str(tmp_path / "missing.mp3")]) == 1
    assert "file not found" in capsys.readouterr().err
