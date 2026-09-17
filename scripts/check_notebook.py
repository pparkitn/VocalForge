#!/usr/bin/env python3
"""Lightweight CI checks for the VocalForge notebook (no GPU / no heavy deps).

Validates:
  1. The notebook is valid nbformat 4 JSON with the expected structure.
  2. Every Python code cell (skipping shell magics) is syntactically valid.
  3. kernel-metadata.json is consistent with the notebook (public, GPU T4 x2,
     the three model datasets attached, 1MB source limit respected).
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "ai-local.ipynb"
META = ROOT / "kernel-metadata.json"

EXPECTED_DATASETS = {
    "dsptlp/auk-base",
    "dsptlp/qwen25-omni-shard13",
    "dsptlp/qwen25-omni-shard2",
}

EXPECTED_TASKS = [
    "zero_shot_tts", "content_edit", "instruct_tts", "pitch_edit",
    "speed_edit", "de_accent", "speech_enhance", "speech_separation",
    "timbre_vc", "lyric_edit", "volume_edit", "emotion_edit",
    "vocal_extraction", "target_speaker_extraction", "nonverbal_remove",
    "nonverbal_add", "content_insert", "content_remove", "emotion_sad",
    "enhance_denoise", "enhance_dereverb", "vocal_all_human",
    "separation_full", "target_speaker_full",
]

errors = []


def fail(msg: str) -> None:
    errors.append(msg)


def check_notebook() -> None:
    if not NB.is_file():
        fail(f"missing {NB}")
        return
    nb = json.loads(NB.read_text())
    if nb.get("nbformat") != 4:
        fail("nbformat is not 4")
    cells = nb.get("cells", [])
    if len(cells) < 10:
        fail(f"unexpectedly few cells ({len(cells)})")

    # 1. banner image is the first cell
    if cells[0].get("cell_type") != "markdown" or "data:image/" not in cells[0]["source"]:
        fail("cell 0 is not the banner image markdown (data:image base64)")
    # 2. capability summary right after
    if cells[1].get("cell_type") != "markdown" or "What AuK can do" not in cells[1]["source"]:
        fail("cell 1 is not the 'What AuK can do' summary")

    # 3. weights are loaded from Kaggle datasets (not huggingface at runtime)
    all_src = "\n".join(c.get("source", "") for c in cells)
    for marker in ("auk_base.safetensors", "model-0000*-of-00003.safetensors"):
        if marker not in all_src:
            fail(f"missing dataset marker glob for {marker}")
    if 'subprocess.run(["hf", "download"' in all_src.replace(" ", ""):
        fail("notebook still downloads weights from Hugging Face at runtime")

    # 4. all 24 task names present
    for task in EXPECTED_TASKS:
        if f'"name": "{task}"' not in all_src:
            fail(f"task {task!r} missing from notebook")

    # 5. every code cell compiles (shell magics skipped)
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        src = cell["source"]
        if any(line.lstrip().startswith("!") for line in src.splitlines()):
            continue  # shell magic cell
        try:
            compile(src, f"ai-local.ipynb:cell-{i}", "exec")
        except SyntaxError as e:
            fail(f"cell {i} syntax error: {e}")


def check_metadata() -> None:
    if not META.is_file():
        fail(f"missing {META}")
        return
    meta = json.loads(META.read_text())
    if meta.get("is_private", True) is not False:
        fail("kernel-metadata.json must keep the notebook public (is_private: false)")
    if meta.get("enable_gpu") is not True:
        fail("kernel-metadata.json must enable the GPU")
    if meta.get("machine_shape") != "NvidiaTeslaT4":
        fail("kernel-metadata.json must request NvidiaTeslaT4 (2x T4)")
    if set(meta.get("dataset_sources", [])) != EXPECTED_DATASETS:
        fail(f"dataset_sources mismatch: {meta.get('dataset_sources')}")
    if "docker_image" in meta:
        fail("kernel-metadata.json must NOT pin docker_image (breaks GPU allocation)")
    if NB.stat().st_size > 1_000_000:
        fail("notebook >1MB: Kaggle rejects kernel sources over 1MB")


def main() -> int:
    check_notebook()
    check_metadata()
    if errors:
        print("CI check failed:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("notebook + metadata checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())