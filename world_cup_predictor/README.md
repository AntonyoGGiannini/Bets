# World Cup Predictor 2026 ⚽

Modelo preditivo para estimar **probabilidades** de resultados em partidas da
Copa do Mundo de 2026.

Para cada jogo, o modelo calcula:

- Probabilidade de vitória do Time A / empate / vitória do Time B
- Probabilidade de over/under gols (0.5, 1.5, 2.5)
- Probabilidade de ambos os times marcarem
- Placar mais provável
- (Opcional) Comparação com odds de mercado e cálculo de *edge*

> O foco **não** é "adivinhar o resultado", e sim produzir uma estimativa
> probabilística **bem calibrada**, comparável às probabilidades implícitas
> do mercado.

---

## Racional do modelo

A Copa do Mundo tem características que tornam a previsão difícil: poucos jogos,
alta variância, seleções que jogam pouco juntas e muito peso de contexto. Por
isso o modelo combina três dimensões:

1. **Força estrutural** da seleção → rating **Elo**.
2. **Forma recente** → médias de gols marcados/sofridos nos últimos N jogos,
   ajustadas pela força dos adversários.
3. **Contexto** da partida → fase do torneio, jogo eliminatório, descanso.

A abordagem é um **modelo híbrido interpretável**:

```
Elo ajustado  →  gols esperados (λ)  →  Poisson  →  Monte Carlo  →  probabilidades
```

Vantagens: é interpretável, funciona com poucos dados, permite simular placares
e gera probabilidades para vários mercados de uma vez.

### Como os λ (gols esperados) são estimados

```
λ_A = média_liga × força_ofensiva_A × fragilidade_defensiva_B × ajuste_elo × ajuste_contexto
λ_B = média_liga × força_ofensiva_B × fragilidade_defensiva_A / ajuste_elo × ajuste_contexto
```

As razões de forma recente são **encolhidas em direção a 1.0** (shrinkage),
porque com ~10 jogos a forma é ruidosa — isso evita previsões extremas e
melhora a calibração. O ajuste de Elo é deliberadamente suave para não
duplicar a informação de força já contida na forma.

Depois de estimar os λ, uma distribuição de **Poisson** gera a matriz de
placares, com a **correção de Dixon-Coles** nos placares baixos (a Poisson
simples trata os gols como independentes e subestima empates 0-0/1-1; o backtest
confirmava empate previsto 23.7% vs. observado 25.7%). Uma simulação de
**Monte Carlo** (10.000 partidas) confirma as probabilidades de forma independente.

---

## Estrutura do projeto

```
world_cup_predictor/
├── data/
│   ├── raw/                  # dados brutos (vazio na V1; usa mock)
│   └── processed/
├── notebooks/
│   └── exploratory_analysis.ipynb
├── src/
│   ├── data_loader.py        # carga de dados + geração de base mock
│   ├── feature_engineering.py# Elo diff, forma recente, contexto, forças
│   ├── elo_model.py          # cálculo/atualização de ratings Elo
│   ├── goal_model.py         # λ esperados + matriz de placares (Poisson)
│   ├── monte_carlo.py        # simulação de partidas/torneio
│   ├── odds_analysis.py      # odds → prob. implícita, edge, classificação
│   ├── evaluation.py         # Brier, Log Loss, calibração, erro de gols
│   ├── pipeline.py           # orquestra a V1 de ponta a ponta
│   ├── backtest.py           # backtest walk-forward + métricas
│   ├── simulate_tournament_2026.py  # Monte Carlo da Copa inteira (chaveamento oficial → campeão)
│   └── top_scorer.py         # candidatos a artilheiro (goalscorers.csv)
├── app/
│   └── streamlit_app.py      # app interativo de demonstração
├── outputs/
│   ├── predictions.csv       # tabela final por partida
│   ├── simulations.csv       # resultados do Monte Carlo
│   ├── odds_edges.csv        # comparação modelo vs. mercado
│   └── backtest_predictions.csv
├── requirements.txt
└── README.md
```

---

## Como rodar localmente

Requer Python 3.10+.

```bash
# 1. (opcional) ambiente virtual
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. instalar dependências
pip install -r requirements.txt

# 3. rodar o pipeline V1 com dados mock (gera os CSVs em outputs/)
python src/pipeline.py

# 3b. (opcional) rodar com DADOS REAIS — baixe o CSV antes (ver seção abaixo)
python src/pipeline.py data/raw/results.csv

# 4. rodar o backtest de calibração (mock)
python src/backtest.py
#     ...ou com dados reais:
python src/backtest.py data/raw/results.csv

# 5. (opcional) app interativo
streamlit run app/streamlit_app.py
```

Para a V1, **não é preciso nenhum dado externo nem API**: o
`data_loader` gera uma base histórica fictícia, porém realista (placares
sorteados de uma Poisson condicionada à força latente de cada seleção).

### Exemplo de saída

```
Brasil x Alemanha  (Grupo)
  lambda Brasil:   1.44
  lambda Alemanha: 0.99
  Vitória Brasil:  47%
  Empate:          27%
  Vitória Alemanha:26%
  Over 2.5:        44%
  Under 2.5:       56%
  Ambos marcam:    48%
  Placar provável: 1x0
```

### Calibração (backtest walk-forward)

O backtest prevê cada jogo usando **apenas** a informação anterior a ele
(sem vazamento), e só depois atualiza o Elo.

**Dados mock (600–800 partidas sintéticas):**

```
Brier  modelo:   0.650   (baseline 33/33/33: 0.667)   ← menor é melhor
LogLoss modelo:  1.074   (baseline 33/33/33: 1.099)
Erro de gols (MAE): ~0.91,  bias ≈ 0
```

**Dados reais (≈2.900 partidas de seleções da Copa 2026, pós-2000):**

```
Brier  modelo:   0.608   (baseline 33/33/33: 0.667)   ← bate o baseline
LogLoss modelo:  1.016   (baseline 33/33/33: 1.099)
Calibração:      Vit A 46.0% prev / 45.1% obs · Empate 25.1% / 25.7% · Vit B 30.1% / 29.2%
```

O modelo **bate o baseline** e está bem calibrado nas três saídas. O ganho de
Brier é modesto — o que é honesto: prever futebol internacional é genuinamente
difícil. A calibração (previsto ≈ observado) é o que mais importa para comparar
com o mercado.

### Por que tantos placares 1x1? (peso do Elo + display)

Uma versão anterior previa **1x1 em 67%** dos jogos da Copa. A causa: o
`ELO_SCALE` (peso da diferença de Elo nos λ) estava baixo demais (0.0005), então
favoritos e azarões saíam com λ quase idênticos (~1.3) — faixa em que a Poisson
tem **modo natural em 1x1**.

A correção (grid search no backtest real): `ELO_SCALE = 0.0011`. Isso separa
melhor os times **e melhora o Brier** (0.609 → 0.608). Os placares 1x1 caíram
para **49%** dos jogos, com favoritos claros mostrando placares realistas
(Brasil×Haiti → 3x0, França×Iraq → 3x0).

Os ~49% restantes são jogos **genuinamente equilibrados**, onde 1x1/1x0 *são* os
placares modais reais. Para não dar falsa impressão de certeza (o placar modal
costuma ter só ~12%), o app mostra o **top-3 de placares** com suas
probabilidades, em vez de um único "placar provável".

### Correção de Dixon-Coles (empates)

A Poisson com gols independentes subestimava sistematicamente os empates
(previsto 23.7% vs. observado 25.7% no backtest). A correção de **Dixon-Coles**
(`goal_model.py`: `DIXON_COLES_RHO`) ajusta as quatro células de baixa pontuação
(0-0, 0-1, 1-0, 1-1). O `rho = -0.12` foi re-calibrado por grid search após o
aumento do `ELO_SCALE` (que reduz empates): mantém empate previsto ≈ 25.1%
(observado 25.7%) com Brier 0.608.

### Vantagem de campo (e por que a Copa é neutra)

Uma primeira versão do backtest real mostrava o modelo **subestimando o
mandante**: previa ~45% de vitória do `time_a` quando o observado era ~54%.
A causa: no histórico real o `time_a` joga em casa nas eliminatórias e
amistosos, e o Elo + forma não capturavam isso sozinhos.

A correção foi um termo explícito de **vantagem de campo** (`goal_model.py`:
`HOME_ADV_ATTACK`, `HOME_ADV_DEFENSE`) aplicado **somente** quando
`mando_neutro == 0`. Resultado: a calibração ficou justa (pred ≈ obs em todas
as faixas) e o Brier caiu de 0.616 para 0.601.

> **A Copa do Mundo é em campo neutro.** Por isso a vantagem de campo é
> **zerada** nas previsões da Copa (`mando_neutro = 1`) — exceto para os
> **anfitriões** (EUA, México, Canadá), que mandam os jogos de grupo em casa.
> Assim o modelo é calibrado no histórico real **sem** injetar um viés de
> mando espúrio nos jogos neutros do mundial.

---

## Usando dados reais (recomendado para a V3)

### Fonte 1 — Histórico de partidas: Kaggle (gratuito)

O dataset mais completo de resultados internacionais:

```bash
pip install kaggle
kaggle datasets download martj42/international-football-results-from-1872-to-2017
unzip international-football-results-from-1872-to-2017.zip -d data/raw/
```

O arquivo `results.csv` (~48.000 partidas desde 1872) é normalizado
automaticamente pelo módulo `data_fetcher`:

```python
from data_fetcher import load_kaggle_results, filter_copa_teams

matches = load_kaggle_results("data/raw/results.csv", min_date="2000-01-01")
matches_copa = filter_copa_teams(matches)  # só seleções da Copa 2026
```

Ou via atalho em `data_loader`:

```python
from data_loader import load_matches_kaggle
matches = load_matches_kaggle("data/raw/results.csv")
```

### Fonte 2 — Odds históricas: football-data.co.uk (gratuito)

CSVs por temporada/torneio disponíveis diretamente (sem API key).
Suporta colunas `B365H/D/A`, `PSH/D/A`, `AvgH/D/A`:

```python
from data_loader import load_odds_footballdata
odds = load_odds_footballdata("data/raw/odds_wc2022.csv")
```

### Fonte 3 — Odds ao vivo: The Odds API

Para as partidas da Copa 2026 em tempo real (free tier: 500 req/mês):
obtenha uma chave em [the-odds-api.com](https://the-odds-api.com) e use
`requests` para buscar os mercados antes de cada rodada.

### Diagnóstico do CSV carregado

```bash
python src/data_fetcher.py data/raw/results.csv
```

Imprime estatísticas: n_partidas, período, gols/time médio, distribuição
por competição.

### Formato mínimo para CSV customizado

Se preferir construir seu próprio CSV:

```
data_jogo, time_a, time_b, gols_time_a, gols_time_b, competicao, fase
```

Valores válidos para `competicao`: `World Cup`, `Continental`, `Qualifiers`,
`Nations League`, `Friendly`.

Colunas extras (`xG`, `chutes`, `dias_descanso`, `ranking_fifa`, etc.) são
bem-vindas — serão aproveitadas nas versões futuras.

---

## Limitações (leia antes de levar a sério)

- **Dados mock na V1.** Os números servem para validar o framework, não para
  apostar. Conecte dados reais antes de qualquer uso prático.
- **Mercado é eficiente.** Odds de Copa são acompanhadas por muitos analistas;
  encontrar *edge* real é difícil. Divergências indicam **o que investigar**,
  não recomendações.
- **Poucos jogos / alta variância.** Seleções jogam pouco; risco de overfitting
  é alto. A forma recente é encolhida justamente por isso.
- **Correlação de gols.** A Poisson simples ignora a correlação em placares
  baixos (0x0, 1x1). Aplicamos a correção **Dixon-Coles** (`rho = -0.12`) que
  recalibra os empates; uma versão completa estimaria `rho` junto com os λ (V4).
- **Amistosos valem menos** que jogos oficiais; o modelo pondera por competição,
  mas a heurística é simples.
- **`edge` não é decisão.** Avalie liquidez, margem da casa, qualidade dos
  dados e a incerteza do próprio modelo.

---

## Roadmap

- **V1 — MVP (este repositório):** Elo + forma recente + Poisson + Monte Carlo,
  com dados mock e tabela de probabilidades. ✅
- **V2 — Odds e edge:** importar odds, remover margem, calcular e classificar
  *edge*. ✅ (já incluído em `odds_analysis.py`)
- **V3 — Dados reais, calibração e backtest:** ingestão de dados reais
  (`data_fetcher.py`, ~48k partidas do Kaggle), backtest walk-forward em
  Copas/Euro/Eliminatórias com Brier/Log Loss, e termo de **vantagem de
  campo** condicionado ao mando (neutro na Copa, exceto anfitriões). ✅
  🟡 **Pendente:** odds reais via football-data.co.uk.
- **V4 — Modelo avançado:** xG, escalações, lesões, valor de elenco,
  **Dixon-Coles completo** (estimar `rho` junto com os λ), XGBoost/LightGBM.

A ordem correta é: **(1) prever probabilidades → (2) medir calibração →
(3) comparar com mercado → (4) só então buscar edge.** Sem calibração, um
modelo sofisticado gera sinais falsos.

---

## Aviso

Projeto educacional/analítico. Não é aconselhamento de apostas. Aposte com
responsabilidade.
