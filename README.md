# 🌍 Análise e Modelagem Preditiva de Conflitos Globais para os Negócios

Um repositório para a dissertação de mestrado sobre risco político e impactos nas empresas, com:
- dados (planilhas e extrações),
- análise (Jupyter/Colab),
- e um dashboard interativo para explorar tudo de forma visual.

---

## 🔗 Atalhos úteis

- [Abrir o notebook principal](results/gpt.ipynb)
- [Executar o dashboard](app/app.py)
- [Ver o fluxograma do método](flowchart/flowchart.PNG)
- [Baixar o conjunto final de previsão](results/previsao_risco_cidades.csv)

---

## 🧭 Visão geral do projeto

O projeto está dividido em três partes práticas:

1) Revisão Sistemática da Literatura (SLR)
   - Coleta, triagem e extração de estudos (Springer, Scopus e Web of Science).
   - Arquivos: bib, csv, PDFs, protocolo SLR.

2) Análise de Dados e Modelagem (Notebook)
   - Limpeza das séries (2016–2025), engenharia de variáveis e risco (Probabilidade x Impacto).
   - Perfis temporais (KMeans) e tendência por capital (Regressão Linear).

3) Dashboard Interativo (Streamlit)
   - Upload do dataset, textos processados com embeddings,
   - clusters, palavras‑chave e gráficos exploratórios.

---

## 🚀 Como executar

Você pode rodar a análise (notebook) e o dashboard (app) de forma independente.

### 1) Análise principal (Notebook)

Arquivo: [results/analise_conflitos.ipynb](results/gpt.ipynb)

Passo a passo:
1. Abra no Google Colab ou Jupyter local.
2. Instale dependências:
   ```
   pip install pandas plotly numpy scikit-learn pycountry openpyxl
   ```
3. Faça upload de [data/gpt.xlsx](data/gpt.xlsx) quando o notebook pedir.
4. Execute as células para obter:
   - gráficos da P1 (Risco = Probabilidade x Impacto),
   - clusters temporais (KMeans, k=5),
   - tendência por capital (Regressão Linear),
   - e o arquivo final [results/previsao_risco_cidades.csv](results/previsao_risco_cidades.csv).

Imagens sugeridas (adicione na pasta assets/):
- ![Exemplo — Mapa de risco](results/Graph1.PNG)
- ![Exemplo — Séries temporais por cluster](results/Graph3.PNG)
- ![Exemplo — Tendência por capital](results/Graph4.PNG)

---

### 2) Dashboard interativo (Streamlit)

Arquivo: [app/app.py](app/app.py)

Pré‑requisitos:
- Tenha [data/gpt.xlsx](data/gpt.xlsx) na pasta data/.

Instalação e execução:
```
pip install -r requirements.txt
streamlit run app/app.py
```

O que o app faz:
- Lê e valida o dataset (colunas e datas).
- Cria embeddings de texto com o modelo sentence-transformers/all-MiniLM-L6-v2 (dimensão 384) via Transformers + PyTorch.
- Aplica TF‑IDF + KMeans para agrupar temas e destacar palavras‑chave.
- Exibe gráficos interativos (Altair e Plotly) para explorar clusters, linhas do tempo e frequências.

---

## 📦 Dependências

Arquivo: [requirements.txt](app/requirements.txt)

```txt
streamlit
pandas
numpy
plotly
altair
scikit-learn
transformers
torch
pycountry
openpyxl
```

Observações diretas:
- Para embeddings: Transformers + Torch.
- Para clustering: scikit‑learn (TF‑IDF, KMeans, TSNE).
- Para gráficos: Plotly e Altair.
- Para Excel: openpyxl.

---

## 🗂️ Estrutura do repositório

Uma visão por pastas e o que você encontra em cada uma.

```
.
├── app/
│   └── app.py                  # App do Streamlit
├── data/
│   └── gpt.xlsx                # Série mensal de conflitos (2016–2025), por país/cidade
├── flowchart/
│   └── flowchart.html          # Fluxograma do método
├── results/
│   ├── analise_conflitos.ipynb # Notebook principal (limpeza, P1, P2, gráficos)
│   └── previsao_risco_cidades.csv
├── slr/
│   ├── data/
│   │   ├── studies.bib
│   │   └── studies.csv
│   ├── prints/                 # Evidências da busca 
│   ├── results/
│   │   ├── data_extraction.xlsx
│   │   ├── articles.xlsx
│   │   └── rsl.pdf             # Protocolo SLR
│   └── studies/                # PDFs dos estudos
└── requirements.txt
```
---

## 🧪 Detalhes da modelagem

- Limpeza:
  - Conversão de datas, normalização de nomes de países/capitais (pycountry),
  - Binarização de variáveis “X” quando necessário.

- P1 — Mapa de Risco:
  - Cálculo direto: Risco = Probabilidade x Impacto (escala padronizada).
  - Saídas: mapas/heatmaps, ranking por país/cidade.

- P2 — Dinâmica no tempo:
  - KMeans em séries normalizadas para perfis de risco (estável, crescente, volátil, etc.).
  - Regressão por capital para estimar tendência (coeficiente anual e p‑valor).
  - Saída: [previsao_risco_cidades.csv](results/previsao_risco_cidades.csv) com perfil, tendência e justificativas.

- Texto e IA no dashboard:
  - Embeddings: sentence-transformers/all-MiniLM-L6-v2 (dimensão 384).
  - TF‑IDF para palavras‑chave por cluster.
  - TSNE opcional para visualização 2D dos embeddings.

---

## 📊 O que você pode explorar

- Quais capitais apresentam tendência de risco crescente no período 2016–2025.
- Como os países se agrupam por padrão temporal (KMeans).
- Quais temas aparecem com mais força nos textos (TF‑IDF) dentro de cada cluster.
- Justificativas e resumos visuais para comunicação clara.

---

## 📝 Notas sobre dados

- [data/gpt.xlsx](data/gpt.xlsx): planilha base com agregação mensal por país/cidade (2016–2025).
- O notebook valida formatos e sinaliza colunas ausentes.
- Se houver novas colunas, mantenha nomes consistentes para reaproveitar os gráficos.

---

## 🤝 Contribuições

- Issues: descreva o que deseja reproduzir (arquivo, célula, gráfico, trecho do app).
- Pull requests: inclua exemplos (print, gif curto) e explique a diferença no resultado.
- Logs ajudam: versão do Python, SO, e comandos usados.

---

## 📚 Citação

Se este trabalho ajudar você, cite assim:

> Sousa, C. A. (2025). Aprendizado de Máquina Aplicada aos Riscos Políticos. Instituto de Ciência da Computação. Universidade de São Paulo.