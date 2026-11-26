#!/usr/bin/env python3
"""Confere se o vocabulário bate entre as três fontes do projeto.

O LibrasFone define a lista de sinais em três lugares, e eles precisam ser o
mesmo conjunto, senão o texto do TCC, a coleta e o treino divergem:

  1. SIGNS, no topo de collect_data.py  (o que a coleta pede para gravar)
  2. as chaves de falas.json            (o que vira voz)
  3. as pastas em dataset/              (o que foi de fato gravado)

Classes que começam com "_" (ex.: _outro, a classe de rejeição) são ignoradas.

Sai com código 1 se houver divergência — pode entrar no CI depois que o
vocabulário for unificado. Só usa a biblioteca padrão: roda em qualquer lugar.
"""
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def ler_signs() -> set[str] | None:
    """Lê SIGNS de collect_data.py sem importar o módulo (evita cv2/mediapipe)."""
    arquivo = ROOT / "collect_data.py"
    if not arquivo.exists():
        return None
    arvore = ast.parse(arquivo.read_text(encoding="utf-8"))
    for no in arvore.body:
        nomeia_signs = isinstance(no, ast.Assign) and any(
            isinstance(alvo, ast.Name) and alvo.id == "SIGNS" for alvo in no.targets
        )
        if nomeia_signs and isinstance(no.value, (ast.List, ast.Tuple)):
            return {
                el.value for el in no.value.elts
                if isinstance(el, ast.Constant) and isinstance(el.value, str)
            }
    return None


def ler_falas() -> set[str] | None:
    arquivo = ROOT / "falas.json"
    if not arquivo.exists():
        return None
    dados = json.loads(arquivo.read_text(encoding="utf-8"))
    return {chave for chave in dados if not chave.startswith("_")}


def ler_dataset() -> set[str] | None:
    pasta = ROOT / "dataset"
    if not pasta.exists():
        return None
    return {
        p.name for p in pasta.iterdir()
        if p.is_dir() and not p.name.startswith("_")
    }


def main() -> int:
    fontes = {
        "SIGNS (collect_data.py)": ler_signs(),
        "falas.json": ler_falas(),
        "dataset/": ler_dataset(),
    }

    faltando = [nome for nome, conj in fontes.items() if conj is None]
    if faltando:
        print("Não consegui ler: " + ", ".join(faltando))
        return 1

    todos = sorted(set().union(*fontes.values()))
    print("Fontes: " + " | ".join(f"{nome}={len(conj)}" for nome, conj in fontes.items()))
    print()

    # Tabela: cada sinal e em quais fontes ele aparece.
    cabecalho = f"{'sinal':<16}" + "".join(f"{nome.split()[0]:<14}" for nome in fontes)
    print(cabecalho)
    print("-" * len(cabecalho))
    divergentes = 0
    for sinal in todos:
        marcas = []
        presente_em = 0
        for conj in fontes.values():
            if sinal in conj:
                marcas.append("sim")
                presente_em += 1
            else:
                marcas.append("—")
        if presente_em != len(fontes):
            divergentes += 1
        linha = f"{sinal:<16}" + "".join(f"{m:<14}" for m in marcas)
        print(linha + ("   <-- DIVERGE" if presente_em != len(fontes) else ""))

    print()
    if divergentes:
        print(f"FALHOU: {divergentes} sinal(is) não está(ão) nas três fontes.")
        print("Unifique o vocabulário (mesma lista em collect_data.py, falas.json e dataset/)")
        print("e alinhe com o número declarado no TCC antes da banca.")
        return 1

    print(f"OK: as três fontes têm o mesmo vocabulário ({len(todos)} sinais).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
