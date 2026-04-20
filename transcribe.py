#!/usr/bin/env python3
"""
Transkribiert eine Audiodatei mit lokalem Whisper und speichert das Ergebnis als .txt-Datei.

Verwendung:
    python transcribe.py <audiodatei> [--modell <modell>]

Modelle (nach Genauigkeit):
    tiny, base, small, medium, large (Standard: large)

Beispiel:
    python transcribe.py aufnahme.mp3
    python transcribe.py aufnahme.mp3 --modell medium
"""

import argparse
import sys
from pathlib import Path

import whisper


def transkribiere(audiodatei: str, modell_name: str = "large") -> None:
    audio_pfad = Path(audiodatei)
    if not audio_pfad.exists():
        print(f"Fehler: Datei '{audiodatei}' nicht gefunden.", file=sys.stderr)
        sys.exit(1)

    ausgabe_pfad = audio_pfad.with_suffix(".txt")

    print(f"Lade Modell '{modell_name}' ...")
    modell = whisper.load_model(modell_name)

    print(f"Transkribiere '{audio_pfad.name}' auf Deutsch ...")
    ergebnis = modell.transcribe(str(audio_pfad), language="de", verbose=False)

    text = ergebnis["text"].strip()

    ausgabe_pfad.write_text(text, encoding="utf-8")
    print(f"Transkription gespeichert: {ausgabe_pfad}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audiodatei mit Whisper auf Deutsch transkribieren"
    )
    parser.add_argument("audiodatei", help="Pfad zur Audiodatei (mp3, wav, m4a, ...)")
    parser.add_argument(
        "--modell",
        default="large",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper-Modell (Standard: large)",
    )
    args = parser.parse_args()
    transkribiere(args.audiodatei, args.modell)


if __name__ == "__main__":
    main()
