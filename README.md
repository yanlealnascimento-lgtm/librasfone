# LibrasFone

Tradutor de Libras em tempo real pra chamadas de vídeo. Projeto que a gente (grupo de 5) desenvolveu como TCC de Engenharia de Software na UCSAL.

A ideia: a câmera captura o sinal que a pessoa faz, o programa reconhece, converte pra texto e fala a tradução por um microfone virtual. Assim dá pra usar em qualquer chamada (Meet, Teams, Zoom) sem precisar instalar nada do outro lado.

**Status:** o fluxo completo está implementado — captura dos pontos da mão (MediaPipe), segmentação por linha de descanso, classificação por MLP com classe de rejeição, síntese de voz local e saída por microfone virtual (VB-Cable). O app principal é o `librasfone.py`, com os modos palavra e frase, registro de sessão (`--registrar`) e teste de acerto por roteiro (`--roteiro`). Os resultados da avaliação estão no documento do TCC.

> **Importante sobre reprodutibilidade:** o modelo treinado, o dataset completo e os registros das sessões de teste **não estão versionados aqui** — veja a seção [Reprodutibilidade](#reprodutibilidade). O que está no Git é o código e uma amostra do dataset.

## Rodando

Precisa de Python 3.14 e, no Windows, do driver **VB-Cable** instalado (é por ele que o áudio chega na chamada).

```
pip install -r requirements.txt
```

Pra ver a câmera com os pontos da mão desenhados:

```
python landmark_viewer.py
```

Aperta `Q` pra fechar. Na primeira vez ele baixa o modelo do MediaPipe, então precisa de internet.

Pra rodar o tradutor você precisa de um **modelo treinado** em `models/librasfone_model.joblib`. Ele não vem no repositório; gere com:

```
python train.py        # lê o dataset/, treina a MLP e salva o modelo
python librasfone.py   # abre o tradutor ao vivo (modo frase por padrão)
```

Opções úteis do `librasfone.py`:

| Flag | O que faz |
|---|---|
| `--modo palavra` / `--modo frase` | uma palavra por vez, ou várias concatenadas numa pausa |
| `--limiar 0.7` | confiança mínima pra aceitar um sinal |
| `--sem-audio` | só mostra o texto, não fala |
| `--testar-audio` | testa a voz e o VB-Cable e sai |
| `--registrar` | grava a sessão (latência, CPU) em `testes/` |
| `--roteiro` | pede cada palavra e anota o que foi reconhecido (liga `--registrar`) |

## Coleta de dados

A gente montou o próprio dataset gravando os sinais. Detalhe importante: **não salva vídeo** — ele guarda só as coordenadas dos pontos da mão em cada frame. Cada amostra é uma sequência de 30 frames (mais ou menos 1-2 segundos de movimento).

```
python collect_data.py
```

| Tecla | Ação |
|---|---|
| `N` / seta direita | próximo sinal |
| `P` / seta esquerda | sinal anterior |
| `ESPAÇO` | grava uma amostra |
| `Q` | sair |

Pra gravar: escolhe o sinal, aperta `ESPAÇO`, espera a contagem (3, 2, 1) e faz o sinal quando a borda ficar vermelha. Repete umas 10-15 vezes por sinal, variando um pouco (distância, ângulo, velocidade) pra não ficar sempre igual.

A lista dos sinais fica na variável `SIGNS` no topo do `collect_data.py`.

## Pra galera do grupo (ler antes de gravar!)

**Todo mundo grava as MESMAS palavras.** Não divide as palavras entre a gente! O modelo precisa ver cada sinal feito por várias pessoas diferentes — se cada palavra tiver só uma pessoa, ele acaba decorando a mão daquela pessoa em vez de aprender o sinal de verdade.

Usa o vídeo do INES (tá na planilha do vocabulário) como referência, pra todo mundo fazer o sinal do mesmo jeito.

Antes de sair gravando, dois passos rápidos:

**1. Põe teu nome no `COLLECTOR_ID`** (topo do `collect_data.py`):

```python
COLLECTOR_ID = "seunome"   # minúsculo, sem espaço nem acento
```

Enquanto tiver vazio ele não deixa gravar. Isso entra no nome dos teus arquivos pra não bater com os dos outros.

**2. Confere a mão:** roda o script, levanta **só a mão direita** e olha se aparece `(D)` na tela. Se aparecer `(E)`, muda `SWAP_HANDEDNESS` pra `False` no topo do arquivo e testa de novo. Faz isso antes de gravar, senão tuas amostras saem com as mãos trocadas.

### Mandando as gravações

Manda **só a pasta `dataset/`**:

```
git pull --rebase
git add dataset/
git commit -m "amostras do seunome"
git push
```

Como cada arquivo tem teu nome, não dá conflito com os dos outros. **Não** commita tuas mudanças no `collect_data.py` (o `COLLECTOR_ID` é só teu), por isso usa `git add dataset/` e não `git add .`. Se der ruim no pull por causa disso, roda `git stash`, `git pull --rebase` e `git stash pop`.

## Formato do dataset (pra quem quiser entender)

Os arquivos ficam em `dataset/<sinal>/<sinal>_<nome>_0001.npy`. Cada um é um array `(30, 126)`: 30 frames × 2 mãos × 21 pontos × (x, y, z). Os primeiros 63 valores são sempre a mão direita, os outros 63 a esquerda; mão que não aparece fica zerada. Cada mão é normalizada pelo punho, então não importa onde ela tá na tela, só a forma e o movimento.

Tem uma cópia sem normalizar em `dataset_raw/` (fica fora do Git) caso a gente precise testar outra normalização mais pra frente. Não apaga a tua.

## Reprodutibilidade

Pra quem for reproduzir ou avaliar o trabalho, o que está e o que não está neste repositório:

**Está versionado:** todo o código do fluxo (`librasfone.py`, `train.py`, `recognition.py`, `segmenter.py`, `features.py`, `tts.py`, `audio_out.py`, `medicao.py`, `roteiro.py`, `painel.py` e os demais), o `requirements.txt` e uma **amostra** do dataset em `dataset/`.

**Não está versionado (gerado ou mantido fora do Git):**

- `models/librasfone_model.joblib` — o modelo treinado. Gere com `python train.py`.
- O **dataset completo** usado na avaliação do TCC (todos os sinalizadores e repetições). A pasta `dataset/` aqui é só uma amostra; a base bruta fica em `dataset_raw/`, fora do Git.
- `dataset/_sinalizadores.json` — mapa de cada vídeo para a pessoa que sinaliza, usado pelo teste de generalização por sinalizador (`train.py`).
- `testes/` — os registros das sessões (`eventos.csv`, `cpu.csv`, `roteiro.csv`, `resumo.txt`) que embasam as latências, o uso de CPU e os acertos relatados no TCC.

Pra reproduzir os números do TCC é preciso restaurar o dataset completo, o `_sinalizadores.json` e, se for conferir as sessões, os registros em `testes/`.

## Próximos passos

- [x] Visualização dos landmarks
- [x] Coleta de dados (2 mãos, em equipe)
- [x] Treino do modelo de reconhecimento (MLP + teste de generalização por sinalizador)
- [x] TTS + microfone virtual (VB-Cable)
- [x] Teste em chamada (Google Meet / Microsoft Teams) — ver resultados no TCC
- [ ] Versionar os artefatos de avaliação (modelo, dataset completo, logs de `testes/`) ou documentar o acesso a eles
- [ ] Melhorar a generalização para sinalizadores de fora do grupo de treino
