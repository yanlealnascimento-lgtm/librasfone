import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from augment import NORMAL, STRONG, augment
from features import FEATURE_SIZE, batch_features
from landmarks import SAMPLE_SHAPE

ROOT = Path(__file__).parent
DATASET_DIR = ROOT / "dataset"
# Liga cada vídeo à pessoa que sinaliza nele. Com esse arquivo, o teste de
# generalização separa as amostras por sinalizador, e não apenas por vídeo.
SIGNERS_FILE = DATASET_DIR / "_sinalizadores.json"
MANIFEST_FILE = DATASET_DIR / "_videos.json"
MODEL_PATH = ROOT / "models" / "librasfone_model.joblib"

TARGET_PER_CLASS = 400
NEGATIVE_MULTIPLIER = 3  # as classes negativas recebem 3x mais amostras: elas cobrem gestos muito variados
MODEL_VERSION = 1


def source_of(path: Path) -> str:
    parts = path.stem.split("_")
    sign_parts = len(path.parent.name.split("_"))
    return "_".join(parts[sign_parts:-1]) or "?"


def load_sample(path: Path) -> np.ndarray:
    sample = np.load(path, allow_pickle=False)
    # Um arquivo .npz com extensão .npy é aberto como NpzFile, e não como array.
    if not isinstance(sample, np.ndarray):
        raise ValueError("o arquivo não contém um único array NumPy")
    if sample.shape != SAMPLE_SHAPE:
        raise ValueError(f"formato {sample.shape}, esperado {SAMPLE_SHAPE}")
    if sample.dtype.kind not in "biuf":  # bool, inteiro ou float
        raise ValueError("as coordenadas devem ser números reais")
    sample = sample.astype(np.float32)
    if not np.isfinite(sample).all():
        raise ValueError("as coordenadas contêm NaN ou infinito")
    return sample


def _signer_of_file() -> dict[str, str]:
    # amostra (palavra/arquivo.npy) -> sinalizador, passando pelo vídeo de origem
    if not SIGNERS_FILE.exists() or not MANIFEST_FILE.exists():
        return {}
    signers = json.loads(SIGNERS_FILE.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    by_video = signers.get("videos", {})
    return {name: by_video[video] for video, entry in manifest.items() if video in by_video
            for name in entry.get("files", [])}


def load_dataset() -> tuple[list[np.ndarray], list[str], list[str]]:
    samples, labels, sources = [], [], []
    signer_of = _signer_of_file()
    bad = []
    for folder in sorted(p for p in DATASET_DIR.iterdir() if p.is_dir()):
        for path in sorted(folder.glob("*.npy")):
            try:
                sample = load_sample(path)
            except (OSError, ValueError, EOFError) as exc:
                bad.append(f"{path} ({exc})")
                continue
            samples.append(sample)
            labels.append(folder.name)
            sources.append(signer_of.get(f"{folder.name}/{path.name}", source_of(path)))
    if bad:
        print("Ignorando amostras inválidas:")
        for b in bad:
            print("  ", b)
    return samples, labels, sources


def expand(samples, labels, rng, cfg, per_class) -> tuple[list[np.ndarray], list[str]]:
    by_class = defaultdict(list)
    for s, y in zip(samples, labels):
        by_class[y].append(s)
    out_x, out_y = [], []
    for y, items in by_class.items():
        out_x += items
        out_y += [y] * len(items)
        target = per_class * NEGATIVE_MULTIPLIER if y.startswith("_") else per_class
        missing = max(target - len(items), 0)
        for i in range(missing):
            out_x.append(augment(items[i % len(items)], rng, cfg))
            out_y.append(y)
    return out_x, out_y


def build_model(seed: int):
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(256, 128),
            alpha=1e-3,
            batch_size=128,
            learning_rate_init=1e-3,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=seed,
        ),
    )


def fit(samples, labels, seed, per_class):
    rng = np.random.default_rng(seed)
    x, y = expand(samples, labels, rng, NORMAL, per_class)
    model = build_model(seed)
    model.fit(batch_features(x), np.array(y))
    return model


def _print_accuracy(y, pred, title):
    print(f"\n{title}: {np.mean(pred == y) * 100:.1f}% de acerto ({len(y)} testes)")
    classes = sorted(set(y))
    per = {c: np.mean(pred[y == c] == c) * 100 for c in classes}
    worst = sorted(per.items(), key=lambda kv: kv[1])[:5]
    print("  piores palavras: " + ", ".join(f"{c} {v:.0f}%" for c, v in worst))
    # Conta os pares (palavra certa, palavra prevista) em que o modelo errou.
    confusions = Counter((a, b) for a, b in zip(y, pred) if a != b)
    if confusions:
        common = sorted(confusions.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:5]
        print("  confusões mais comuns: " + ", ".join(f"{a}->{b} ({n})" for (a, b), n in common))


def accuracy_report(model, samples, labels, cfg, rng, copies=20, title=""):
    x, y = [], []
    for s, label in zip(samples, labels):
        for _ in range(copies):
            x.append(augment(s, rng, cfg))
            y.append(label)
    pred = model.predict(batch_features(x))
    _print_accuracy(np.array(y), pred, title)


def leave_one_source_out(samples, labels, sources, seed, per_class, signers_available):
    sources_by_label = defaultdict(set)
    for label, source in zip(labels, sources):
        sources_by_label[label].add(source)

    def testable(i: int) -> bool:
        label = labels[i]
        return not label.startswith("_") and len(sources_by_label[label]) >= 2

    test_sources = sorted({sources[i] for i in range(len(samples)) if testable(i)})
    if not test_sources:
        print("\n(Teste deixando uma fonte de fora: é preciso ter pelo menos 2 fontes diferentes")
        print(" na mesma palavra. Quando houver mais vídeos por palavra, ele rodará automaticamente.)")
        return

    hits = total = 0
    for src in test_sources:
        # A classe de rejeição fica sempre no treino: sem ela, o modelo não teria
        # exemplos de gestos fora do vocabulário quando a fonte deles é a testada.
        train_idx = [i for i in range(len(samples)) if sources[i] != src or labels[i].startswith("_")]
        test_idx = [i for i in range(len(samples)) if sources[i] == src and testable(i)]
        model = fit([samples[i] for i in train_idx], [labels[i] for i in train_idx], seed, per_class)
        pred = model.predict(batch_features([samples[i] for i in test_idx]))
        ok = sum(p == labels[i] for p, i in zip(pred, test_idx))
        hits += ok
        total += len(test_idx)
        print(f"  sem '{src}': acertou {ok}/{len(test_idx)}")
    rotulo = "por sinalizador" if signers_available else "por fonte do arquivo"
    print(f"Teste deixando uma fonte de fora ({rotulo}): {hits / total * 100:.1f}% ({hits}/{total})")
    if not signers_available:
        print(
            "\n  *** ATENÇÃO: dataset/_sinalizadores.json ausente. As 'fontes' foram\n"
            "  inferidas do nome do arquivo, NÃO da pessoa que sinaliza. Este número\n"
            "  mede generalização entre ARQUIVOS, não entre PESSOAS, e NÃO deve ser\n"
            "  reportado no TCC como generalização por sinalizador. Gere o\n"
            "  _sinalizadores.json (ou rode com --exigir-sinalizadores) antes de citá-lo. ***"
        )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Treina o modelo do LibrasFone.")
    parser.add_argument("--por-palavra", type=int, default=TARGET_PER_CLASS,
                        help="amostras por palavra no treino, contando as variações (padrão 400)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sem-teste-fonte", action="store_true",
                        help="pula o teste deixando uma fonte de fora (mais rápido)")
    parser.add_argument("--exigir-sinalizadores", action="store_true",
                        help="exige dataset/_sinalizadores.json para o teste de generalização; "
                             "recusa treinar se o mapa de sinalizadores não existir")
    return parser


def _save_model(model, counts):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "labels": list(model.classes_),
            "feature_size": FEATURE_SIZE,
            "version": MODEL_VERSION,
            "counts": dict(counts),
            "trained_at": time.strftime("%Y-%m-%d %H:%M"),
        },
        MODEL_PATH,
        compress=3,
    )


def main() -> int:
    parser = _argument_parser()
    args = parser.parse_args()
    if args.por_palavra < 1:
        parser.error("--por-palavra deve ser positivo")
    # O scikit-learn só aceita random_state entre 0 e 2**32 - 1. Sem esta
    # verificação, um valor fora do intervalo só daria erro depois de gerar as variações.
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("--seed deve estar entre 0 e 4294967295")

    if not DATASET_DIR.exists():
        print("A pasta dataset/ não existe. Rode extract_videos.py ou collect_data.py antes.")
        return 1
    samples, labels, sources = load_dataset()
    counts = Counter(labels)
    words = [c for c in counts if not c.startswith("_")]
    if len(words) < 2:
        print("São necessárias pelo menos 2 palavras com amostras para treinar.")
        return 1

    print(f"{len(samples)} amostras reais, {len(counts)} classes:")
    for label in sorted(counts):
        n_src = len({s for s, y in zip(sources, labels) if y == label})
        extra = "  (não é falada)" if label.startswith("_") else ""
        print(f"  {label:15s} {counts[label]:3d} amostra(s) de {n_src} fonte(s){extra}")

    start = time.time()
    model = fit(samples, labels, args.seed, args.por_palavra)
    print(f"\nTreino concluído em {time.time() - start:.0f}s")

    rng = np.random.default_rng(args.seed + 1)
    accuracy_report(model, samples, labels, NORMAL, rng, title="Variações normais")
    accuracy_report(model, samples, labels, STRONG, rng, title="Variações fortes")
    signers_available = SIGNERS_FILE.exists() and MANIFEST_FILE.exists()
    if args.exigir_sinalizadores and not signers_available:
        print("\nErro: --exigir-sinalizadores foi pedido, mas falta dataset/_sinalizadores.json "
              "(ou _videos.json).", file=sys.stderr)
        print("O teste de generalização por sinalizador não pode ser feito sem esse mapa.", file=sys.stderr)
        return 1
    if not args.sem_teste_fonte:
        print()
        leave_one_source_out(samples, labels, sources, args.seed, args.por_palavra, signers_available)

    _save_model(model, counts)
    print(f"\nModelo salvo em {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")
    if min(counts[w] for w in words) < 5:
        print("\nAviso: há palavras com menos de 5 amostras reais. O modelo funciona, mas")
        print("generaliza muito melhor com mais vídeos de pessoas diferentes para cada palavra.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
