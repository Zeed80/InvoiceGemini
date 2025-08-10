#!/usr/bin/env python
"""
Python utility to generate (.ts) and compile (.qm) Qt translation files.

- Scans the project sources (app/**/*.py, app/**/*.ui, debug_runner.py)
- Uses pylupdate6/pyside6-lupdate/lupdate to produce .ts files
- Uses lrelease/pyside6-lrelease to produce .qm files

Usage examples:
  python generate_translations.py --languages ru,en
  python generate_translations.py --pyqt-bin-path "C:\\Path\\To\\Python\\Scripts" --languages ru,en
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from typing import Iterable, List, Optional


def find_sources(project_root: str) -> List[str]:
    sources: List[str] = []
    app_dir = os.path.join(project_root, "app")
    for root, _, files in os.walk(app_dir):
        for name in files:
            if name.endswith(".py") or name.endswith(".ui"):
                sources.append(os.path.join(root, name))

    runner = os.path.join(project_root, "debug_runner.py")
    if os.path.exists(runner):
        sources.append(runner)
    return sources


def resolve_tool(candidates: Iterable[str], pyqt_bin_path: Optional[str]) -> Optional[str]:
    # 1) Try explicit path if provided
    if pyqt_bin_path:
        for name in candidates:
            candidate = os.path.join(pyqt_bin_path, name)
            if os.path.exists(candidate):
                return candidate

    # 2) Try PATH
    for name in candidates:
        path = shutil.which(name)
        if path:
            return path
    return None


def write_pro_file(translations_dir: str, sources: List[str]) -> str:
    pro_path = os.path.join(translations_dir, "translations.pro")
    with open(pro_path, "w", encoding="utf-8") as f:
        f.write("SOURCES += \\\n")
        for i, src in enumerate(sources):
            normalized = src.replace("\\", "/")
            cont = " \\\n" if i < len(sources) - 1 else "\n"
            f.write(f"    {normalized}{cont}")
    return pro_path


def run(cmd: List[str]) -> None:
    print("> ", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.stdout:
        print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}: {' '.join(cmd)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and compile Qt translations (.ts -> .qm)")
    parser.add_argument("--languages", "-l", required=False, default="ru,en",
                        help="Comma-separated list of language codes, e.g. ru,en,de")
    parser.add_argument("--pyqt-bin-path", required=False, default=None,
                        help="Optional path to directory containing pylupdate6/lrelease tools")
    args = parser.parse_args()

    languages = [lang.strip() for lang in args.languages.split(",") if lang.strip()]
    if not languages:
        print("No languages specified")
        return 2

    project_root = os.getcwd()
    translations_dir = os.path.join(project_root, "translations")
    os.makedirs(translations_dir, exist_ok=True)

    print(f"📂 Project root: {project_root}")
    print(f"🌐 Translations dir: {translations_dir}")

    sources = find_sources(project_root)
    if not sources:
        print("No sources found in app/ or debug_runner.py")
        return 1
    print(f"📝 Sources found: {len(sources)} files")

    pro_path = write_pro_file(translations_dir, sources)
    print(f"🗂  Using .pro file: {pro_path}")

    pylupdate_candidates = [
        "pylupdate6.exe", "pylupdate6",
        "pyside6-lupdate.exe", "pyside6-lupdate",
        "lupdate.exe", "lupdate",
    ]
    lrelease_candidates = [
        "lrelease.exe", "lrelease",
        "pyside6-lrelease.exe", "pyside6-lrelease",
    ]

    lupdate = resolve_tool(pylupdate_candidates, args.pyqt_bin_path)
    lrelease = resolve_tool(lrelease_candidates, args.pyqt_bin_path)

    if not lupdate:
        print("ERROR: pylupdate/lupdate not found. Provide --pyqt-bin-path or add the tools to PATH.")
        return 2
    if not lrelease:
        print("ERROR: lrelease not found. Provide --pyqt-bin-path or add the tools to PATH.")
        return 2

    print(f"🛠 lupdate:  {lupdate}")
    print(f"🛠 lrelease: {lrelease}")

    # Generate TS per language
    ts_files: List[str] = []
    for lang in languages:
        ts_out = os.path.join(translations_dir, f"invoicegemini_{lang}.ts")
        ts_files.append(ts_out)
        print(f"📝 Generating TS for '{lang}': {ts_out}")
        run([lupdate, pro_path, "-ts", ts_out])

    # Compile every TS -> QM
    for ts in ts_files:
        qm = ts[:-3] + ".qm"
        print(f"🛠 Compiling QM: {qm}")
        run([lrelease, ts, "-qm", qm])
        if not os.path.exists(qm):
            raise RuntimeError(f"Failed to create {qm}")

    print(f"✅ Done. .qm files are in: {translations_dir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)


