# LibrasFone

Tradutor de Libras em tempo real pra chamadas de vídeo. Projeto que a gente (grupo de 5) tá desenvolvendo.

A ideia: a câmera captura o sinal que a pessoa faz, o programa reconhece, converte pra texto e fala a tradução por um microfone virtual. Assim dá pra usar em qualquer chamada (Meet, Teams, Zoom) sem precisar instalar nada do outro lado.

**Status:** em desenvolvimento. Por enquanto temos a captura dos pontos da mão (MediaPipe) e a ferramenta de coleta de dados funcionando. O reconhecimento e o áudio ainda vêm por aí.

## Rodando

Precisa de Python 3.14.

```
pip install -r requirements.txt
```

Pra ver a câmera com os pontos da mão desenhados:

```
python landmark_viewer.py
```

Aperta `Q` pra fechar. Na primeira vez ele baixa o modelo do MediaPipe, então precisa de internet.

## Coleta de dados

A gente tá montando o próprio dataset gravando os sinais. Detalhe importante: **não salva vídeo** — ele guarda só as coordenadas dos pontos da mão em cada frame. Cada amostra é uma sequência de 30 frames (mais ou menos 1-2 segundos de movimento).

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

A lista dos 29 sinais fica na variável `SIGNS` no topo do `collect_data.py`.

## Pra galera do grupo (ler antes de gravar!)

**Todo mundo grava as MESMAS 29 palavras.** Não divide as palavras entre a gente! O modelo precisa ver cada sinal feito por várias pessoas diferentes — se cada palavra tiver só uma pessoa, ele acaba decorando a mão daquela pessoa em vez de aprender o sinal de verdade.

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

## Próximos passos

- [x] Visualização dos landmarks
- [x] Coleta de dados (2 mãos, em equipe)
- [ ] Juntar o dataset do grupo
- [ ] Treinar o modelo de reconhecimento
- [ ] TTS + microfone virtual
- [ ] Testar numa chamada de verdade