import argparse
import hashlib
import json
import re
import sys
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

from camera_io import valid_fps, video_timestamp
from landmark_viewer import create_landmarker, detect_hands, draw_hands
from landmarks import extract_hands, hand_size, presence, to_sample
from segmenter import RAISE_HAND_SIZES, top_of_hands

ROOT = Path(__file__).parent
VIDEOS_DIR = ROOT / "videos"
DATASET_DIR = ROOT / "dataset"
MANIFEST = DATASET_DIR / "_videos.json"
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}

MIN_SEGMENT_FRAMES = 10  # trechos mais curtos são ignorados
MAX_GAP_FRAMES = 6       # falhas curtas (mão abaixa ou some por poucos frames) não dividem um trecho
PAD_FRAMES = 3           # frames extras mantidos antes e depois de cada trecho

REST_TIME = 0.3          # segundos do começo e do fim usados como pose de descanso
REST_MIN_TOP = 0.7       # no descanso as mãos ficam baixas (abaixo de 70% da altura da imagem)
MOVE_HAND_SIZES = 0.8    # quanto a mão precisa sair da pose de descanso para contar como sinal
MAX_GAP_TIME = 0.4       # pausas mais curtas que isso (s) não dividem um sinal
LOOP_RAISE = 2.5         # num loop a mão quase não sobe em relação ao começo/fim do vídeo
LOOP_WINDOW = 1.5        # duração (s) de cada amostra cortada de um vídeo em loop
LOOP_STEP = 0.75         # passo (s) entre essas amostras
KEY_POINTS = [0, 4, 8, 9, 12, 16, 20]  # pulso, base do dedo médio e pontas dos dedos


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "", text.lower())
    return slug or "video"


def file_hash(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):  # lê em blocos de 1 MB
            h.update(chunk)
    return h.hexdigest()[:16]


@contextmanager
def _open_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"não foi possível abrir {path}")
        yield cap
    finally:
        cap.release()


def _detections(cap, landmarker, fps: float):
    index = 0
    last_timestamp = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # O timestamp vem da posição do frame no vídeo, e não do relógio.
        timestamp = video_timestamp(index / fps, last_timestamp)
        last_timestamp = timestamp
        yield index, frame, detect_hands(landmarker, frame, timestamp)
        index += 1


def read_landmarks(path: Path) -> tuple[np.ndarray, float]:
    frames = []
    with _open_video(path) as cap, create_landmarker() as landmarker:
        fps = valid_fps(cap.get(cv2.CAP_PROP_FPS))
        for _, frame, result in _detections(cap, landmarker, fps):
            height, width = frame.shape[:2]
            hands, _ = extract_hands(result, width / height)
            frames.append(hands)
    if not frames:
        raise RuntimeError(f"{path} não tem frames")
    return np.stack(frames), fps


def find_segments(frames: np.ndarray, fps: float = 30.0) -> list[tuple[int, int]]:
    """Encontra os trechos do vídeo em que a pessoa está sinalizando.

    Há três tipos de vídeo:
    - a pessoa começa ou termina com as mãos baixas e paradas (descanso): o
      sinal é o trecho em que as mãos sobem ou saem dessa pose;
    - as mãos ficam fora da imagem quando a pessoa não sinaliza: o sinal é o
      trecho em que elas aparecem;
    - o vídeo já começa e termina no meio do sinal, repetindo-o (loop): ele é
      cortado em janelas curtas, e cada uma vira uma amostra.
    """
    total = len(frames)
    if total == 0:
        return []
    present = presence(frames)
    seen = present.any(axis=1)
    if seen.sum() < MIN_SEGMENT_FRAMES:
        return []

    sizes = [hand_size(frames[t, s]) for t in range(total) for s in range(2) if present[t, s]]
    size = max(float(np.median(sizes)), 1e-3)
    tops = [top_of_hands(frames[t], present[t]) for t in range(total)]
    tops = np.array([np.nan if v is None else v for v in tops])
    edges = _edge_frames(present, fps)
    rest = [ref for ref in edges if np.nanmedian(tops[ref]) >= REST_MIN_TOP]

    if _is_loop(tops, edges, seen, size):
        idx = np.flatnonzero(seen)
        return _loop_windows(int(idx[0]), int(idx[-1]) + 1, fps)

    if rest:
        rest_top = max(float(np.nanmedian(tops[ref])) for ref in rest)
        raised = np.nan_to_num(tops, nan=np.inf) < rest_top - RAISE_HAND_SIZES * size
        moved = _distance_from_rest(frames, present, rest, size) > MOVE_HAND_SIZES
        active = raised | moved
        if not active.any():
            active = seen
    else:
        active = seen

    max_gap = max(MAX_GAP_FRAMES, int(MAX_GAP_TIME * fps))
    return [
        (max(0, a - PAD_FRAMES), min(total, b + PAD_FRAMES))
        for a, b in _active_spans(active, max_gap)
        if b - a >= MIN_SEGMENT_FRAMES
    ]


def _is_loop(tops, edges, seen, size) -> bool:
    # O vídeo começa com a mão alta (no meio do sinal), as mãos aparecem o tempo
    # todo e nunca sobem muito acima das poses do começo e do fim.
    if len(edges) < 2 or seen.mean() <= 0.9:
        return False
    start_top = float(np.nanmedian(tops[edges[0]]))
    lowest_edge = max(float(np.nanmedian(tops[ref])) for ref in edges)
    return start_top < REST_MIN_TOP and (lowest_edge - np.nanmin(tops)) / size < LOOP_RAISE


def _edge_frames(present: np.ndarray, fps: float) -> list[np.ndarray]:
    # Frames com mãos no começo e no fim do vídeo (só conta se a maioria tiver mãos).
    total = len(present)
    k = min(max(3, int(REST_TIME * fps)), total)
    refs = []
    for part in (np.arange(0, k), np.arange(total - k, total)):
        with_hands = part[present[part].any(axis=1)]
        if len(with_hands) >= k // 2:
            refs.append(with_hands)
    return refs


def _distance_from_rest(frames, present, rest, size) -> np.ndarray:
    # Para cada frame, o quanto as mãos estão longe da pose de descanso, em
    # tamanhos de mão. Uma mão que aparece sem estar no descanso conta como longe.
    total = len(frames)
    distance = np.zeros(total)
    for t in range(total):
        if not present[t].any():
            continue
        best = np.inf
        for ref in rest:
            far = 0.0
            for s in range(2):
                if not present[t, s]:
                    continue
                ref_frames = ref[present[ref, s]]
                if len(ref_frames) == 0:
                    far = np.inf
                    continue
                pose = np.median(frames[ref_frames, s][:, KEY_POINTS, :2], axis=0)
                gap = np.linalg.norm(frames[t, s, KEY_POINTS, :2] - pose, axis=1).mean() / size
                far = max(far, gap)
            best = min(best, far)
        distance[t] = best
    return distance


def _loop_windows(first: int, last: int, fps: float) -> list[tuple[int, int]]:
    window = max(MIN_SEGMENT_FRAMES, int(LOOP_WINDOW * fps))
    step = max(1, int(LOOP_STEP * fps))
    if last - first <= window:
        return [(first, last)]
    return [(a, a + window) for a in range(first, last - window + 1, step)]


def _active_spans(active: np.ndarray, max_gap: int = MAX_GAP_FRAMES) -> list[tuple[int, int]]:
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
            if gap > max_gap:
                segments.append((start, t - gap + 1))
                start, gap = None, 0
    if start is not None:
        segments.append((start, len(active) - gap))
    return segments


def load_manifest() -> dict:
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError(f"{MANIFEST}: o manifesto deve ser um objeto JSON")
        return manifest
    return {}


def save_manifest(manifest: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(MANIFEST)


def _output_path(name: str) -> Path:
    relative = Path(name)
    root = DATASET_DIR.resolve()
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root) or path.suffix != ".npy":
        raise ValueError(f"caminho de amostra inválido no manifesto: {name}")
    return path


def _cached_outputs_exist(entry: dict) -> bool:
    files = entry.get("files")
    if not isinstance(files, list):
        return False
    try:
        return all(_output_path(name).is_file() for name in files)
    except (TypeError, ValueError):
        return False


def remove_outputs(entry: dict) -> None:
    for name in entry.get("files", []):
        _output_path(name).unlink(missing_ok=True)


def unique_prefix(sign: str, video: Path) -> str:
    digest = hashlib.sha1(str(video.relative_to(VIDEOS_DIR)).encode()).hexdigest()[:4]
    return f"{sign}_{slugify(video.stem)}{digest}"


def save_samples(sign: str, video: Path, frames: np.ndarray, segments: list[tuple[int, int]]) -> list[str]:
    prefix = unique_prefix(sign, video)
    files = []
    for n, (a, b) in enumerate(segments, 1):
        name = f"{sign}/{prefix}_{n:04d}.npy"
        path = DATASET_DIR / name
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, to_sample(frames[a:b]))
        files.append(name)
    return files


def show_segments(path: Path, frames_count: int, segments: list[tuple[int, int]], title: str) -> None:
    try:
        with _open_video(path) as cap, create_landmarker() as landmarker:
            fps = valid_fps(cap.get(cv2.CAP_PROP_FPS))
            for index, frame, result in _detections(cap, landmarker, fps):
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
                if cv2.waitKey(max(1, int(1000 / fps))) & 0xFF in (ord("q"), ord("Q")):
                    break
    finally:
        cv2.destroyAllWindows()


def list_videos() -> list[tuple[str, Path]]:
    items = []
    if not VIDEOS_DIR.exists():
        return items
    for folder in sorted(p for p in VIDEOS_DIR.iterdir() if p.is_dir()):
        for video in sorted(folder.rglob("*")):
            if video.is_file() and video.suffix.lower() in VIDEO_EXTS:
                items.append((folder.name, video))
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description="Extrai amostras de sinais a partir de vídeos.")
    parser.add_argument("--refazer", action="store_true", help="extrai novamente todos os vídeos")
    parser.add_argument("--mostrar", action="store_true", help="mostra os trechos encontrados em cada vídeo")
    args = parser.parse_args()

    videos = list_videos()
    if not videos:
        print(f"Nenhum vídeo encontrado. Coloque os vídeos em {VIDEOS_DIR}\\<palavra>\\arquivo.mp4")
        return 1

    try:
        manifest = load_manifest()
    except (OSError, ValueError) as exc:
        print(f"Erro ao ler o manifesto: {exc}", file=sys.stderr)
        return 1
    total_new = 0
    problems = []

    for sign, video in videos:
        if not re.fullmatch(r"_?[a-z0-9_]+", sign):
            problems.append(f"{video}: nome de pasta '{sign}' inválido (use letras minúsculas, sem acentos nem espaços)")
            continue
        key = video.relative_to(VIDEOS_DIR).as_posix()
        entry = manifest.get(key)
        if not isinstance(entry, dict):
            entry = {}
        try:
            digest = file_hash(video)
            if entry.get("hash") == digest and not args.refazer and _cached_outputs_exist(entry):
                continue  # vídeo já processado e sem alterações
            print(f"[{sign}] {video.name} ... ", end="", flush=True)
            frames, fps = read_landmarks(video)
            segments = find_segments(frames, fps)
            files = save_samples(sign, video, frames, segments)
            manifest[key] = {
                "hash": digest,
                "files": files,
                "segments": [[a / fps, b / fps] for a, b in segments],
            }
            save_manifest(manifest)
        except Exception as exc:
            print("ERRO")
            problems.append(f"{video}: {exc}")
            continue

        # Se o vídeo mudou e agora gera menos trechos, apaga as amostras que sobraram.
        try:
            obsolete = [name for name in entry.get("files", []) if name not in files]
            remove_outputs({"files": obsolete})
        except (OSError, TypeError, ValueError) as exc:
            problems.append(f"{video}: falha ao limpar amostras antigas: {exc}")

        if not segments:
            print("nenhum trecho com mãos encontrado")
            problems.append(f"{video}: a pessoa sinalizando não foi encontrada (mãos não detectadas?)")
            continue

        total_new += len(files)
        spans = ", ".join(f"{a / fps:.1f}s-{b / fps:.1f}s" for a, b in segments)
        print(f"{len(segments)} trecho(s): {spans}")

        if args.mostrar:
            show_segments(video, len(frames), segments, f"{sign}/{video.name}")

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
