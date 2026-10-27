import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from landmark_viewer import create_landmarker, draw_hands
from landmarks import SAMPLE_SHAPE, extract_hands, hand_size, to_sample

ROOT = Path(__file__).parent
VIDEOS_DIR = ROOT / "videos"
DATASET_DIR = ROOT / "dataset"
MANIFEST = DATASET_DIR / "_videos.json"
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}

MIN_SEGMENT_FRAMES = 10
MAX_GAP_FRAMES = 6
PAD_FRAMES = 3
RAISE_HAND_SIZES = 1.3


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "", text.lower())
    return slug or "video"


def file_hash(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def read_landmarks(path: Path) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"não consegui abrir {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    index = 0
    with create_landmarker() as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(image, int(index * 1000 / fps))
            hands, _ = extract_hands(result, width / height)
            frames.append(hands)
            index += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"{path} não tem frames")
    return np.stack(frames), fps


def find_segments(frames: np.ndarray) -> list[tuple[int, int]]:
    present = np.any(frames != 0, axis=(2, 3))
    total = len(frames)
    top = np.full(total, np.nan)
    sizes = []
    for t in range(total):
        ys = [frames[t, s, :, 1].min() for s in range(2) if present[t, s]]
        if ys:
            top[t] = min(ys)
        sizes += [hand_size(frames[t, s]) for s in range(2) if present[t, s]]

    if np.isnan(top).all():
        return []

    size = float(np.median(sizes)) if sizes else 0.1
    rest = float(np.nanpercentile(top, 90))
    threshold = rest - RAISE_HAND_SIZES * size
    active = np.nan_to_num(top, nan=np.inf) < threshold

    # vídeo já cortado só com o sinal: usa tudo que tem mão
    if active.mean() > 0.9 or not active.any():
        idx = np.flatnonzero(present.any(axis=1))
        return [(int(idx[0]), int(idx[-1]) + 1)] if len(idx) >= MIN_SEGMENT_FRAMES else []

    segments = []
    start = None
    gap = 0
    for t, on in enumerate(active):
        if on:
            if start is None:
                start = t
            gap = 0
        elif start is not None:
            gap += 1
            if gap > MAX_GAP_FRAMES:
                segments.append((start, t - gap + 1))
                start, gap = None, 0
    if start is not None:
        segments.append((start, total - gap))

    padded = []
    for a, b in segments:
        if b - a < MIN_SEGMENT_FRAMES:
            continue
        padded.append((max(0, a - PAD_FRAMES), min(total, b + PAD_FRAMES)))
    return padded


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {}


def save_manifest(manifest: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def remove_outputs(entry: dict) -> None:
    for name in entry.get("files", []):
        path = DATASET_DIR / name
        if path.exists():
            path.unlink()


def unique_prefix(sign: str, video: Path) -> str:
    # o hash evita colisão entre vídeos com o mesmo nome em pastas diferentes
    digest = hashlib.sha1(str(video.relative_to(VIDEOS_DIR)).encode()).hexdigest()[:4]
    return f"{sign}_{slugify(video.stem)}{digest}"


def show_segments(path: Path, frames_count: int, segments: list[tuple[int, int]], title: str) -> None:
    cap = cv2.VideoCapture(str(path))
    index = 0
    with create_landmarker() as landmarker:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(index * 1000 / fps)
            )
            draw_hands(frame, result)
            current = [n for n, (a, b) in enumerate(segments, 1) if a <= index < b]
            if current:
                cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (0, 0, 255), 10)
                text = f"TRECHO {current[0]}"
            else:
                text = "descanso"
            cv2.putText(frame, f"{title} | {text} | frame {index}/{frames_count}", (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("LibrasFone - trechos (Q pula)", frame)
            if cv2.waitKey(int(1000 / fps)) & 0xFF in (ord("q"), ord("Q")):
                break
            index += 1
    cap.release()
    cv2.destroyAllWindows()


def list_videos() -> list[tuple[str, Path]]:
    items = []
    if not VIDEOS_DIR.exists():
        return items
    for folder in sorted(p for p in VIDEOS_DIR.iterdir() if p.is_dir()):
        for video in sorted(folder.rglob("*")):
            if video.suffix.lower() in VIDEO_EXTS:
                items.append((folder.name, video))
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description="Extrai amostras de sinais a partir de vídeos.")
    parser.add_argument("--refazer", action="store_true", help="extrai de novo todos os vídeos")
    parser.add_argument("--mostrar", action="store_true", help="mostra os trechos encontrados em cada vídeo")
    args = parser.parse_args()

    videos = list_videos()
    if not videos:
        print(f"Nenhum vídeo encontrado. Coloque os vídeos em {VIDEOS_DIR}\\<palavra>\\arquivo.mp4")
        return 1

    manifest = load_manifest()
    total_new = 0
    problems = []

    for sign, video in videos:
        if not re.fullmatch(r"_?[a-z0-9_]+", sign):
            problems.append(f"{video}: nome da pasta '{sign}' inválido (use minúsculas, sem acento/espaço)")
            continue
        key = video.relative_to(VIDEOS_DIR).as_posix()
        digest = file_hash(video)
        entry = manifest.get(key)
        if entry and entry.get("hash") == digest and not args.refazer:
            continue
        if entry:
            remove_outputs(entry)

        print(f"[{sign}] {video.name} ... ", end="", flush=True)
        try:
            frames, fps = read_landmarks(video)
        except Exception as exc:
            print("ERRO")
            problems.append(f"{video}: {exc}")
            continue

        segments = find_segments(frames)
        if not segments:
            print("nenhum trecho com mãos encontrado")
            problems.append(f"{video}: não achei a pessoa sinalizando (mãos não detectadas?)")
            manifest[key] = {"hash": digest, "files": [], "segments": []}
            continue

        prefix = unique_prefix(sign, video)
        files = []
        for n, (a, b) in enumerate(segments, 1):
            sample = to_sample(frames[a:b])
            assert sample.shape == SAMPLE_SHAPE
            out = DATASET_DIR / sign / f"{prefix}_{n:04d}.npy"
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out, sample)
            files.append(out.relative_to(DATASET_DIR).as_posix())

        manifest[key] = {
            "hash": digest,
            "files": files,
            "segments": [[a / fps, b / fps] for a, b in segments],
        }
        save_manifest(manifest)
        total_new += len(files)
        spans = ", ".join(f"{a / fps:.1f}s-{b / fps:.1f}s" for a, b in segments)
        print(f"{len(segments)} trecho(s): {spans}")

        if args.mostrar:
            show_segments(video, len(frames), segments, f"{sign}/{video.name}")

    save_manifest(manifest)

    counts = {}
    for path in DATASET_DIR.glob("*/*.npy"):
        counts[path.parent.name] = counts.get(path.parent.name, 0) + 1
    print(f"\n{total_new} amostra(s) nova(s). Amostras por palavra no dataset:")
    for sign in sorted(counts):
        print(f"  {sign:15s} {counts[sign]}")
    if problems:
        print("\nProblemas:")
        for p in problems:
            print(f"  - {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
