"""Unit tests for the sidecar helpers (no models, no network)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yue2_trainer.sidecars import (assemble_style, clean_segments, heuristic_sections, parse_stem,  # noqa: E402
                                   plain_to_sections)


def test_parse_stem():
    assert parse_stem("01 - Daft Punk - Get Lucky (Official Audio)") == ("Daft Punk", "Get Lucky")
    assert parse_stem("Artist_-_Title") == ("Artist", "Title")
    assert parse_stem("my_demo_take3") == (None, "my demo take3")


def test_heuristic_sections_marks_repeats_as_chorus():
    chunks = [(0.0, 2.0, "walking down the road"), (2.0, 4.0, "thinking of you"),
              (8.0, 10.0, "hold on tonight"), (10.0, 12.0, "hold on till the light"),
              (20.0, 22.0, "another day is gone"), (22.0, 24.0, "and i carry on"),
              (30.0, 32.0, "hold on tonight"), (32.0, 34.0, "hold on till the light")]
    out = heuristic_sections(chunks)
    assert out.count("[Chorus]") == 2 and out.count("[Verse]") == 2
    assert out.index("[Verse]") < out.index("[Chorus]")
    assert "hold on tonight" in out


def test_heuristic_sections_intro_and_empty():
    assert heuristic_sections([]) == ""
    out = heuristic_sections([(15.0, 17.0, "late start")])
    assert out.startswith("[Intro]")


def test_plain_to_sections():
    text = "line a\nline b\n\nchorus x\nchorus y\n\nline c\nline d\n\nchorus x\nchorus y\n"
    out = plain_to_sections(text)
    assert out.count("[Chorus]") == 2 and out.count("[Verse]") == 2


def test_assemble_style():
    tags = {"genre": ["city pop", "disco"], "mood": ["upbeat", "groovy"], "vocal": ["female vocal"],
            "instruments": ["synthesizer", "bass guitar"]}
    s = assemble_style("ja", tags, 118)
    assert s == "Japanese, city pop, disco, upbeat, groovy, female vocal, synthesizer, bass guitar, 118 BPM"
    assert assemble_style(None, {"genre": ["Pop", "pop"]}, None) == "Pop"


def test_clean_segments_drops_hallucinations():
    chunks = [(0.0, 1.0, "作曲 李宗盛")] * 10 + [(20.0, 22.0, "real line"), (22.0, 24.0, "real line"),
                                              (24.0, 26.0, "real line"), (30.0, 32.0, "Subtitles by the Amara.org community"),
                                              (40.0, 42.0, "another line")]
    kept, dropped = clean_segments(chunks)
    texts = [t for _, _, t in kept]
    assert texts == ["real line", "real line", "another line"]
    assert dropped == 12


def test_fuzzy_chorus_detection():
    chunks = [(0.0, 2.0, "walking down the road"), (2.0, 4.0, "thinking of you"),
              (10.0, 12.0, "hold on tonight my love"), (12.0, 14.0, "hold on until the light"),
              (20.0, 22.0, "another day is gone"), (22.0, 24.0, "and i carry on"),
              (30.0, 32.0, "hold on tonight, my love"), (32.0, 34.0, "hold on until the lights")]
    out = heuristic_sections(chunks)
    assert out.count("[Chorus]") == 2
