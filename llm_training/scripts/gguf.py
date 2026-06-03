"""
Convert an already-merged HuggingFace model to GGUF and quantize it to Q4.

Examples:
    python llm_training/scripts/gguf.py
    python llm_training/scripts/gguf.py --merged-dir llm_training/merged_model_v3
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
TRAINING_DIR = ROOT_DIR / "llm_training"

DEFAULT_MERGED_DIR = TRAINING_DIR / "merged_model_v3"
DEFAULT_LLAMA_CPP_DIR = TRAINING_DIR / "llama.cpp"
DEFAULT_OUTPUT_DIR = TRAINING_DIR / "gguf"
DEFAULT_F16_GGUF = DEFAULT_OUTPUT_DIR / "driving-mentor-v3-f16.gguf"
DEFAULT_Q4_GGUF = DEFAULT_OUTPUT_DIR / "driving-mentor-v3-q4_k_m.gguf"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+ " + " ".join(str(part) for part in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_command(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required command not found: {name}")


def ensure_llama_cpp(args: argparse.Namespace) -> Path:
    llama_cpp_dir = Path(args.llama_cpp_dir)
    if not llama_cpp_dir.exists():
        if args.no_clone:
            raise FileNotFoundError(f"llama.cpp not found: {llama_cpp_dir}")
        ensure_command("git")
        run([
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/ggerganov/llama.cpp",
            str(llama_cpp_dir),
        ])

    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        raise FileNotFoundError(f"convert_hf_to_gguf.py not found: {convert_script}")

    if not args.skip_requirements:
        requirements = llama_cpp_dir / "requirements.txt"
        if requirements.exists():
            run([sys.executable, "-m", "pip", "install", "-r", str(requirements)])

    return llama_cpp_dir


def find_quantize_bin(llama_cpp_dir: Path) -> Path | None:
    names = {"llama-quantize", "llama-quantize.exe", "quantize", "quantize.exe"}
    for path in (llama_cpp_dir / "build").rglob("*"):
        if path.name in names and path.is_file():
            return path
    return None


def build_quantize(llama_cpp_dir: Path, args: argparse.Namespace) -> Path:
    existing = find_quantize_bin(llama_cpp_dir)
    if existing and not args.rebuild:
        print(f"quantize binary found: {existing}")
        return existing

    ensure_command("cmake")
    build_dir = llama_cpp_dir / "build"
    run([
        "cmake",
        "-B",
        str(build_dir),
        "-S",
        str(llama_cpp_dir),
        "-DGGML_CUDA=OFF",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=OFF",
    ])
    run([
        "cmake",
        "--build",
        str(build_dir),
        "--target",
        "llama-quantize",
        f"-j{args.jobs}",
    ])

    quantize_bin = find_quantize_bin(llama_cpp_dir)
    if not quantize_bin:
        raise FileNotFoundError("llama-quantize build finished, but binary was not found")
    if os.name != "nt":
        quantize_bin.chmod(0o755)
    print(f"quantize binary built: {quantize_bin}")
    return quantize_bin


def convert_to_f16(args: argparse.Namespace, llama_cpp_dir: Path) -> Path:
    merged_dir = Path(args.merged_dir)
    f16_gguf = Path(args.f16_gguf)
    if not merged_dir.exists():
        raise FileNotFoundError(f"Merged model directory not found: {merged_dir}")

    f16_gguf.parent.mkdir(parents=True, exist_ok=True)
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    run([
        sys.executable,
        str(convert_script),
        str(merged_dir),
        "--outtype",
        "f16",
        "--outfile",
        str(f16_gguf),
    ])
    return f16_gguf


def quantize_q4(args: argparse.Namespace, quantize_bin: Path, f16_gguf: Path) -> Path:
    q4_gguf = Path(args.q4_gguf)
    q4_gguf.parent.mkdir(parents=True, exist_ok=True)
    run([
        str(quantize_bin),
        str(f16_gguf),
        str(q4_gguf),
        args.quant_type,
    ])
    return q4_gguf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert merged HF model to GGUF and quantize to Q4")
    parser.add_argument("--merged-dir", type=Path, default=DEFAULT_MERGED_DIR)
    parser.add_argument("--llama-cpp-dir", type=Path, default=DEFAULT_LLAMA_CPP_DIR)
    parser.add_argument("--f16-gguf", type=Path, default=DEFAULT_F16_GGUF)
    parser.add_argument("--q4-gguf", type=Path, default=DEFAULT_Q4_GGUF)
    parser.add_argument("--quant-type", default="Q4_K_M")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--keep-f16", action="store_true")
    parser.add_argument("--skip-requirements", action="store_true")
    parser.add_argument("--no-clone", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    llama_cpp_dir = ensure_llama_cpp(args)
    quantize_bin = build_quantize(llama_cpp_dir, args)
    f16_gguf = convert_to_f16(args, llama_cpp_dir)
    q4_gguf = quantize_q4(args, quantize_bin, f16_gguf)

    if not args.keep_f16:
        f16_gguf.unlink(missing_ok=True)
        print(f"removed intermediate f16 GGUF: {f16_gguf}")

    print(f"Q4 GGUF ready: {q4_gguf}")


if __name__ == "__main__":
    main()
