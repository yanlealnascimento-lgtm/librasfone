import math
import re
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from landmark_viewer import CAMERA_INDEX, create_landmarker, draw_hands

# coloque seu nome aqui antes de gravar (minúsculo, sem espaço/acento, ex: "testee")
# ele vai no nome dos arquivos pra não dar conflito entre a equipe
COLLECTOR_ID = ""
COLLECTOR_ID_PATTERN = re.compile(r"[a-z0-9]+")

# lista dos sinais (sem acento pq vira nome de pasta)
SIGNS = [
    "como", "voce", "eu", "nome", "qual", "onde", "quando", "quem", "o_que",
    "quero", "preciso", "pode", "nao_pode", "esperar", "repetir", "falar",
    "ouvir", "entender", "nao_entendi", "pergunta", "resposta", "certo",
    "errado", "sim", "nao", "agora", "depois", "por_favor", "obrigado",
]

SEQUENCE_LENGTH = 30
NUM_LANDMARKS = 21
HAND_FEATURES = NUM_LANDMARKS * 3  # x, y, z -> 63 valores por mão

# 0..62 = mão direita, 63..125 = mão esquerda (se a mão não aparecer fica tudo zero)
HAND_SLOTS = ("Right", "Left")
HAND_SLOT_NAMES = ("D", "E")
FRAME_FEATURES = len(HAND_SLOTS) * HAND_FEATURES
SAMPLE_SHAPE = (SEQUENCE_LENGTH, FRAME_FEATURES)

# o mediapipe acha que a imagem tá espelhada, então as mãos vêm trocadas.
# se aparecer (E) quando levantar a mão direita, troca pra False
SWAP_HANDEDNESS = True

COUNTDOWN_SECONDS = 3
# menos frames com mão que isso = amostra descartada
MIN_FRAMES_WITH_HAND = 10
MESSAGE_SECONDS = 2.5

DATASET_DIR = Path(__file__).parent / "dataset"
# cópia sem normalizar, caso a gente queira testar outra normalização depois
SAVE_RAW = True
RAW_DATASET_DIR = Path(__file__).parent / "dataset_raw"

WINDOW_NAME = "LibrasFone - Coleta de dados (Q para sair)"
KEY_SPACE = 32
# setas no Windows
KEY_LEFT = 0x250000
KEY_RIGHT = 0x270000

GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)


def extract_frame_features(result) -> tuple[np.ndarray, np.ndarray, list[bool]]:
    normalized = np.zeros((len(HAND_SLOTS), NUM_LANDMARKS, 3), dtype=np.float32)
    raw = np.zeros_like(normalized)
    present = [False] * len(HAND_SLOTS)

    hands = []
    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        category = handedness[0]
        label = category.category_name
        if SWAP_HANDEDNESS:
            label = "Left" if label == "Right" else "Right"
        hands.append((category.score, label, landmarks))

    # se as duas vierem com o mesmo lado, a de maior score fica e a outra vai pro slot livre
    for _, label, landmarks in sorted(hands, key=lambda hand: hand[0], reverse=True):
        slot = HAND_SLOTS.index(label) if label in HAND_SLOTS else 0
        if present[slot]:
            slot = 1 - slot
        if present[slot]:
            continue

        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
        raw[slot] = points
        # normaliza pelo punho (landmark 0), assim a posição na tela não importa
        normalized[slot] = points - points[0]
        present[slot] = True

    return normalized.reshape(-1), raw.reshape(-1), present


def collector_id_problem() -> str | None:
    if not COLLECTOR_ID:
        return "COLLECTOR_ID vazio"
    if not COLLECTOR_ID_PATTERN.fullmatch(COLLECTOR_ID):
        return f"COLLECTOR_ID invalido: {COLLECTOR_ID!r}"
    return None


def report_collector_id_problem(problem: str) -> None:
    print(
        f"\nAVISO: {problem}.\n"
        'Abra collect_data.py e defina COLLECTOR_ID no topo com o seu nome, ex.: COLLECTOR_ID = "etse"\n'
        "(só letras minúsculas e números, sem espaços nem acentos).\n"
        "A gravação fica BLOQUEADA até isso ser feito.\n",
        file=sys.stderr,
    )


def sample_files(sign: str) -> list[Path]:
    return sorted((DATASET_DIR / sign).glob(f"{sign}_*.npy"))


def own_sample_files(sign: str) -> list[Path]:
    return sorted((DATASET_DIR / sign).glob(f"{sign}_{COLLECTOR_ID}_*.npy"))


def find_incompatible_samples() -> list[tuple[Path, str]]:
    # procura amostras do formato antigo
    incompatible = []
    for root in (DATASET_DIR, RAW_DATASET_DIR):
        for path in sorted(root.rglob("*.npy")):
            try:
                shape = np.load(path, mmap_mode="r").shape
            except Exception:
                incompatible.append((path, "ilegível"))
                continue
            if shape != SAMPLE_SHAPE:
                incompatible.append((path, f"shape {shape}"))
    return incompatible


def report_incompatible_samples(incompatible: list[tuple[Path, str]]) -> None:
    print(
        f"\nAVISO: {len(incompatible)} amostra(s) em formato incompatível com o atual {SAMPLE_SHAPE}.\n"
        "O formato mudou de 63 valores por frame (uma mão) para 126 (duas mãos);\n"
        "amostras antigas não podem ser misturadas com as novas.",
        file=sys.stderr,
    )
    for path, reason in incompatible[:10]:
        print(f"  {path} ({reason})", file=sys.stderr)
    if len(incompatible) > 10:
        print(f"  ... e mais {len(incompatible) - 10}", file=sys.stderr)
    print(
        "A gravação fica BLOQUEADA até você mover ou apagar essas amostras, por exemplo\n"
        "renomeando dataset/ e dataset_raw/ para dataset_antigo/ e dataset_raw_antigo/.\n",
        file=sys.stderr,
    )


def next_sample_index(sign: str) -> int:
    # pega o maior número + 1, pra nunca repetir nome
    pattern = re.compile(rf"{re.escape(sign)}_{re.escape(COLLECTOR_ID)}_(\d+)\.npy")
    used = [0]
    for folder in (DATASET_DIR / sign, RAW_DATASET_DIR / sign):
        for path in folder.glob(f"{sign}_{COLLECTOR_ID}_*.npy"):
            match = pattern.fullmatch(path.name)
            if match:
                used.append(int(match.group(1)))
    return max(used) + 1


def save_sample(sign: str, normalized: np.ndarray, raw: np.ndarray) -> Path:
    name = f"{sign}_{COLLECTOR_ID}_{next_sample_index(sign):04d}.npy"
    targets = [(DATASET_DIR / sign / name, normalized)]
    if SAVE_RAW:
        targets.append((RAW_DATASET_DIR / sign / name, raw))
    for path, data in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        # "x" dá erro se o arquivo já existir, pra não sobrescrever
        with open(path, "xb") as f:
            np.save(f, data)
    return targets[0][0]


def put_text(frame, text, origin, scale=0.7, color=WHITE, thickness=2) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def put_centered_text(frame, text, y, scale, color, thickness) -> None:
    (width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    put_text(frame, text, ((frame.shape[1] - width) // 2, y), scale, color, thickness)


def main() -> int:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Erro: não foi possível abrir a webcam.", file=sys.stderr)
        return 1

    incompatible = find_incompatible_samples()
    if incompatible:
        report_incompatible_samples(incompatible)

    collector_problem = collector_id_problem()
    if collector_problem:
        report_collector_id_problem(collector_problem)

    counts = {sign: 0 if collector_problem else len(own_sample_files(sign)) for sign in SIGNS}
    total_counts = {sign: len(sample_files(sign)) for sign in SIGNS}
    sign_index = 0

    state = "idle"  # idle -> countdown -> recording
    countdown_end = 0.0
    normalized_frames: list[np.ndarray] = []
    raw_frames: list[np.ndarray] = []
    frames_with_hand = 0

    message, message_color, message_until = "", GREEN, 0.0

    start = time.monotonic()
    last_timestamp_ms = -1

    try:
        with create_landmarker(num_hands=len(HAND_SLOTS)) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Erro: falha ao ler frame da webcam.", file=sys.stderr)
                    return 1

                now = time.monotonic()
                # o modo VIDEO precisa de timestamp sempre crescente
                timestamp_ms = max(int((now - start) * 1000), last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                normalized, raw, hands_present = extract_frame_features(result)
                has_hand = any(hands_present)

                sign = SIGNS[sign_index]

                if state == "countdown" and now >= countdown_end:
                    state = "recording"
                    normalized_frames, raw_frames, frames_with_hand = [], [], 0

                if state == "recording":
                    normalized_frames.append(normalized)
                    raw_frames.append(raw)
                    frames_with_hand += has_hand
                    if len(normalized_frames) == SEQUENCE_LENGTH:
                        state = "idle"
                        message_until = now + MESSAGE_SECONDS
                        if frames_with_hand < MIN_FRAMES_WITH_HAND:
                            message = f"DESCARTADA: maos em so {frames_with_hand}/{SEQUENCE_LENGTH} frames"
                            message_color = RED
                        else:
                            path = save_sample(sign, np.stack(normalized_frames), np.stack(raw_frames))
                            counts[sign] += 1
                            total_counts[sign] += 1
                            message = f"Salva: {path.name} (maos em {frames_with_hand}/{SEQUENCE_LENGTH} frames)"
                            message_color = GREEN
                            print(f"Amostra salva em {path}")

                draw_hands(frame, result)
                # espelha só pra mostrar na tela
                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]

                put_text(frame, f"Sinal [{sign_index + 1}/{len(SIGNS)}]: {sign}", (10, 30), 0.9, YELLOW)
                put_text(
                    frame,
                    f"Amostras gravadas: {counts[sign]} suas / {total_counts[sign]} no total",
                    (10, 62),
                )
                if has_hand:
                    sides = "+".join(name for name, here in zip(HAND_SLOT_NAMES, hands_present) if here)
                    put_text(frame, f"Mãos detectadas: {sum(hands_present)} ({sides})", (10, 92), 0.6, GREEN)
                else:
                    put_text(frame, "Maos detectadas: 0", (10, 92), 0.6, RED)

                if incompatible:
                    put_text(frame, f"FORMATO ANTIGO: {len(incompatible)} amostra(s) incompativeis", (10, 126), 0.6, RED)
                    put_text(frame, "Gravacao bloqueada: mova ou apague dataset/ e dataset_raw/", (10, 150), 0.5, RED)
                    put_text(frame, "(lista no console) e abra o script de novo", (10, 172), 0.5, RED)
                elif collector_problem:
                    put_text(frame, f"{collector_problem}: gravacao bloqueada", (10, 126), 0.6, RED)
                    put_text(frame, 'Defina COLLECTOR_ID = "seunome" no topo de collect_data.py', (10, 150), 0.5, RED)
                    put_text(frame, "(minusculas, sem espacos nem acentos) e abra o script de novo", (10, 172), 0.5, RED)

                if state == "countdown":
                    remaining = max(1, math.ceil(countdown_end - now))
                    put_centered_text(frame, str(remaining), height // 2 + 40, 5.0, YELLOW, 10)
                    put_centered_text(frame, "Prepare o sinal...", height // 2 + 90, 0.8, YELLOW, 2)
                elif state == "recording":
                    cv2.rectangle(frame, (0, 0), (width - 1, height - 1), RED, 8)
                    put_centered_text(
                        frame, f"GRAVANDO {len(normalized_frames)}/{SEQUENCE_LENGTH}", height - 50, 0.9, RED, 2
                    )
                    progress = int((width - 40) * len(normalized_frames) / SEQUENCE_LENGTH)
                    cv2.rectangle(frame, (20, height - 35), (width - 20, height - 20), WHITE, 1)
                    cv2.rectangle(frame, (20, height - 35), (20 + progress, height - 20), RED, -1)
                else:
                    if now < message_until:
                        put_text(frame, message, (10, height - 45), 0.6, message_color)
                    put_text(frame, "ESPACO: gravar   N/P ou setas: trocar sinal   Q: sair", (10, height - 15), 0.5)

                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKeyEx(1)
                char = chr(key & 0xFF).lower() if 0 <= key < 256 else ""
                if char == "q":
                    break
                # só deixa trocar de sinal/gravar quando não tá gravando
                if state == "idle":
                    if char == "n" or key == KEY_RIGHT:
                        sign_index = (sign_index + 1) % len(SIGNS)
                    elif char == "p" or key == KEY_LEFT:
                        sign_index = (sign_index - 1) % len(SIGNS)
                    elif key == KEY_SPACE and incompatible:
                        message = "Bloqueado: ha amostras em formato antigo (veja o console)"
                        message_color, message_until = RED, now + MESSAGE_SECONDS
                    elif key == KEY_SPACE and collector_problem:
                        message = "Bloqueado: defina COLLECTOR_ID no topo de collect_data.py"
                        message_color, message_until = RED, now + MESSAGE_SECONDS
                    elif key == KEY_SPACE:
                        state = "countdown"
                        countdown_end = now + COUNTDOWN_SECONDS
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
