"""Create .style.txt / .lyrics.txt sidecars for every song in a folder (no ComfyUI needed).

    python prepare_dataset.py D:/songs
    python prepare_dataset.py D:/songs --lyrics lrclib+whisper --sections claude --overwrite

Lyrics come from LRCLIB (by artist/title tags or "Artist - Title" file names) with Whisper as
fallback; style prompts come from CLAP zero-shot tags + tempo/key + detected language.
Existing sidecars are kept unless --overwrite is given, so you can hand-edit and rerun.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from yue2_trainer.sidecars import DEFAULT_CLAUDE, DEFAULT_WHISPER, SidecarConfig, prepare_folder, summarize  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder")
    p.add_argument("--lyrics", default="lrclib+whisper", choices=["lrclib+whisper", "lrclib", "whisper", "none"])
    p.add_argument("--whisper-model", default=DEFAULT_WHISPER)
    p.add_argument("--language", default="auto", help="Whisper language code (en, zh, ja, ...) or auto.")
    p.add_argument("--no-separate", action="store_true", help="Transcribe the full mix instead of the demucs vocal stem.")
    p.add_argument("--style", default="clap", choices=["clap", "none"])
    p.add_argument("--default-style", default="", help="Style text when --style none.")
    p.add_argument("--sections", default="heuristic", choices=["heuristic", "claude", "none"],
                   help="How to add [Verse]/[Chorus] tags. 'claude' needs ANTHROPIC_API_KEY (or ant auth login).")
    p.add_argument("--claude-model", default=DEFAULT_CLAUDE)
    p.add_argument("--device", default="auto")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-recursive", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = SidecarConfig(lyrics_source=args.lyrics, whisper_model=args.whisper_model, style_source=args.style,
                        language=args.language, separate_vocals=not args.no_separate,
                        default_style=args.default_style, section_tags=args.sections, claude_model=args.claude_model,
                        overwrite=args.overwrite, device=args.device)

    def progress(done, total, name):
        logging.info("[%d/%d] %s", done, total, name)

    reports = prepare_folder(args.folder, cfg, recursive=not args.no_recursive, progress=progress)
    print(summarize(reports))
    return 0


if __name__ == "__main__":
    sys.exit(main())
