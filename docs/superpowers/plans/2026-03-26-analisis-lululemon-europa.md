# Analisis Lululemon Europa - Plan de Implementacion

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Crear un analisis exhaustivo multi-plataforma (Instagram, TikTok, Trustpilot) que justifique y guie una estrategia de marketing para Lululemon en Europa, incluyendo limpieza de datos, EDA, analisis de sentimiento con BERT multilingual, comparativa US vs Europa, y benchmarking contra ALO Yoga.

**Architecture:** Pipeline analitico en 6 notebooks secuenciales. Un modulo `src/` con funciones reutilizables. Datos limpios en `datos/clean/`. Figuras y tablas exportadas a `outputs/` para Word y PPT.

**Tech Stack:** Python 3.11, pandas, matplotlib, seaborn, plotly, transformers (nlptown/bert-base-multilingual-uncased-sentiment), wordcloud, scikit-learn, langdetect

---

## Estructura de archivos

```
analitica-social-final/
├── datos/
│   ├── raw/                          # (existente)
│   ├── processed/                    # (existente)
│   └── clean/                        # Datos normalizados post-limpieza
│       ├── trustpilot_all.csv
│       ├── ig_publicaciones.csv
│       ├── ig_comentarios.csv
│       ├── tiktok_videos.csv
│       └── tiktok_comentarios.csv
├── notebooks/
│   ├── extraccion/                   # Notebooks de scraping (mover existentes)
│   │   ├── Extracción de los datos con API.ipynb
│   │   ├── Extracción de datos Instragram.ipynb
│   │   ├── Extracción Instagram Lululemon.ipynb
│   │   ├── Extracción TikTok Lululemon por Hashtags.ipynb
│   │   └── Extracción Trustpilot Lululemon vs ALO.ipynb
│   └── analisis/
│       ├── 01_limpieza_datos.ipynb
│       ├── 02_EDA.ipynb
│       ├── 03_analisis_sentimiento.ipynb
│       ├── 04_comparativa_us_vs_europa.ipynb
│       ├── 05_competencia_lululemon_vs_alo.ipynb
│       └── 06_insights_estrategia.ipynb
├── src/
│   ├── __init__.py
│   ├── cleaning.py                   # Funciones de limpieza y normalizacion
│   ├── plotting.py                   # Estilo visual uniforme, helpers de graficos
│   └── sentiment.py                  # Wrapper BERT multilingual
├── outputs/
│   ├── figuras/                      # PNG exportados para Word/PPT
│   └── tablas/                       # CSV resumen
├── docs/
├── ppts/
└── requirements.txt
```

---

## Task 0: Setup del repo y dependencias

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `src/cleaning.py`
- Create: `src/plotting.py`
- Create: `src/sentiment.py`
- Create: `datos/clean/` (directorio)
- Create: `outputs/figuras/` (directorio)
- Create: `outputs/tablas/` (directorio)
- Create: `notebooks/extraccion/` (directorio)
- Move: notebooks de extraccion existentes a `notebooks/extraccion/`

- [ ] **Step 1: Crear requirements.txt**

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
plotly>=5.15
wordcloud>=1.9
scikit-learn>=1.3
langdetect>=1.0.9
transformers>=4.30
torch>=2.0
openpyxl>=3.1
```

- [ ] **Step 2: Crear estructura de directorios**

```bash
mkdir -p datos/clean outputs/figuras outputs/tablas notebooks/extraccion notebooks/analisis src
```

- [ ] **Step 3: Mover notebooks de extraccion**

```bash
cd notebooks
mv "Extracción de los datos con API .ipynb" extraccion/
mv "Extracción de datos Instragram.ipynb" extraccion/
mv "Extracción Instagram Lululemon.ipynb" extraccion/
mv "Extracción TikTok Lululemon por Hashtags.ipynb" extraccion/
mv "Extracción Trustpilot Lululemon vs ALO.ipynb" extraccion/
```

- [ ] **Step 4: Crear src/__init__.py**

```python
"""Modulo de utilidades para el analisis de Lululemon."""
```

- [ ] **Step 5: Crear src/plotting.py**

```python
"""Estilo visual uniforme para todos los notebooks."""
import matplotlib.pyplot as plt
import seaborn as sns

# Paleta Lululemon
COLORS = {
    "lululemon_red": "#C8102E",
    "lululemon_dark": "#1D1D1B",
    "alo_green": "#4A7C59",
    "us": "#3B82F6",
    "europe": "#EF4444",
    "es": "#FACC15",
    "fr": "#3B82F6",
    "gb": "#EF4444",
    "de": "#1D1D1B",
    "positive": "#22C55E",
    "neutral": "#F59E0B",
    "negative": "#EF4444",
}

PLATFORM_COLORS = {
    "instagram": "#E1306C",
    "tiktok": "#000000",
    "trustpilot": "#00B67A",
}

def setup_style():
    """Configura estilo global para matplotlib."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
    })

def save_fig(fig, name, subdir="figuras"):
    """Guarda figura en outputs/."""
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "outputs", subdir, f"{name}.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    print(f"Guardado: {path}")
```

- [ ] **Step 6: Crear src/cleaning.py**

```python
"""Funciones de limpieza y normalizacion de datos."""
import pandas as pd
import numpy as np
import re
from langdetect import detect, LangDetectException


def clean_text(text):
    """Limpia texto: elimina URLs, emojis excesivos, normaliza espacios."""
    if pd.isna(text) or str(text).strip() == "":
        return ""
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", "", text)  # URLs
    text = re.sub(r"\s+", " ", text).strip()       # espacios multiples
    return text


def safe_detect_lang(text):
    """Detecta idioma de forma segura."""
    try:
        if pd.isna(text) or len(str(text).strip()) < 10:
            return "unknown"
        return detect(str(text))
    except LangDetectException:
        return "unknown"


def normalize_dates(series, unit=None):
    """Normaliza columna de fechas a datetime."""
    if unit:
        return pd.to_datetime(series, unit=unit, errors="coerce")
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_localize(None)


def map_region(lang_code):
    """Mapea codigo de idioma a region para analisis."""
    mapping = {
        "en": "US/GB",
        "es": "ES",
        "fr": "FR",
        "de": "DE",
    }
    return mapping.get(lang_code, "Other")


def add_engagement_rate(df, likes_col="likes", comments_col="comments", views_col=None):
    """Calcula engagement rate."""
    if views_col and views_col in df.columns:
        df["engagement_rate"] = ((df[likes_col] + df[comments_col]) / df[views_col].replace(0, np.nan)) * 100
    else:
        df["engagement_rate"] = df[likes_col] + df[comments_col] * 3  # score ponderado
    return df
```

- [ ] **Step 7: Crear src/sentiment.py**

```python
"""Analisis de sentimiento con BERT multilingual."""
import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch


_sentiment_pipe = None


def get_sentiment_pipeline():
    """Carga el pipeline de sentimiento (singleton)."""
    global _sentiment_pipe
    if _sentiment_pipe is None:
        model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        _sentiment_pipe = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            device=0 if torch.cuda.is_available() else (
                0 if torch.backends.mps.is_available() else -1
            ),
            truncation=True,
            max_length=512,
        )
    return _sentiment_pipe


def predict_sentiment(texts, batch_size=32):
    """
    Predice sentimiento para una lista de textos.
    Devuelve DataFrame con columnas: stars (1-5), label (negativo/neutro/positivo), score (confianza).
    """
    pipe = get_sentiment_pipeline()

    # Filtrar textos vacios
    clean_texts = []
    indices = []
    for i, t in enumerate(texts):
        t_clean = str(t).strip() if pd.notna(t) else ""
        if len(t_clean) >= 3:
            clean_texts.append(t_clean[:512])
            indices.append(i)

    # Predecir en batches
    results = []
    for i in range(0, len(clean_texts), batch_size):
        batch = clean_texts[i:i+batch_size]
        preds = pipe(batch)
        results.extend(preds)

    # Construir DataFrame de resultados
    stars_list = [np.nan] * len(texts)
    score_list = [np.nan] * len(texts)

    for idx, result in zip(indices, results):
        star = int(result["label"].split()[0])  # "1 star" -> 1
        stars_list[idx] = star
        score_list[idx] = result["score"]

    df_result = pd.DataFrame({
        "sentiment_stars": stars_list,
        "sentiment_confidence": score_list,
    })

    # Clasificar en categorias
    df_result["sentiment_label"] = pd.cut(
        df_result["sentiment_stars"],
        bins=[0, 2, 3, 5],
        labels=["negativo", "neutro", "positivo"],
    )

    return df_result
```

- [ ] **Step 8: Instalar dependencias**

```bash
pip install -r requirements.txt
```

- [ ] **Step 9: Commit**

```bash
git add requirements.txt src/ notebooks/extraccion/
git commit -m "chore: setup repo structure with src modules and move extraction notebooks"
```

---

## Task 1: Notebook 01 - Limpieza de Datos

**Files:**
- Create: `notebooks/analisis/01_limpieza_datos.ipynb`
- Output: `datos/clean/*.csv`

Este notebook carga todos los datos de `datos/processed/`, los limpia, normaliza y guarda en `datos/clean/` con un esquema unificado.

- [ ] **Step 1: Celda 0 - Titulo (markdown)**

```markdown
# 01. Limpieza y Normalizacion de Datos

Cargamos los datos extraidos de Instagram, TikTok y Trustpilot, los limpiamos y normalizamos para crear un dataset unificado listo para analisis.

**Fuentes de datos:**
- Instagram: 585 publicaciones + 2.413 comentarios (US + Europa)
- TikTok: 400 videos + 1.056 comentarios (US + Europa)
- Trustpilot: 383 reviews (Lululemon + ALO Yoga)
```

- [ ] **Step 2: Celda 1 - Setup e imports**

```python
import sys
sys.path.insert(0, "../../src")

import pandas as pd
import numpy as np
from cleaning import clean_text, safe_detect_lang, normalize_dates, add_engagement_rate
from plotting import setup_style
setup_style()

DATA_PROC = "../../datos/processed"
DATA_CLEAN = "../../datos/clean"

import os
os.makedirs(DATA_CLEAN, exist_ok=True)
```

- [ ] **Step 3: Celda 2 - Cargar Trustpilot (markdown + code)**

```markdown
## 1. Trustpilot Reviews
```

```python
tp_lulu = pd.read_csv(f"{DATA_PROC}/trustpilot/trustpilot_lululemon.csv")
tp_alo = pd.read_csv(f"{DATA_PROC}/trustpilot/trustpilot_alo_yoga.csv")

tp_lulu["brand"] = "Lululemon"
tp_alo["brand"] = "ALO Yoga"

tp = pd.concat([tp_lulu, tp_alo], ignore_index=True)

# Limpieza
tp["text_clean"] = tp["text"].apply(clean_text)
tp["title_clean"] = tp["title"].apply(clean_text)
tp["published_date"] = pd.to_datetime(tp["published_date"], errors="coerce", utc=True).dt.tz_localize(None)
tp["experience_date"] = pd.to_datetime(tp["experience_date"], errors="coerce", utc=True).dt.tz_localize(None)
tp["year"] = tp["published_date"].dt.year
tp["month"] = tp["published_date"].dt.to_period("M").astype(str)

# Detectar idioma si no esta
if "language" not in tp.columns or tp["language"].isna().sum() > 0:
    tp["language"] = tp["text_clean"].apply(safe_detect_lang)

# Normalizar pais
tp["country"] = tp["reviewer_country"].str.upper().str.strip()

print(f"Trustpilot total: {len(tp)} reviews")
print(f"  Lululemon: {len(tp[tp['brand']=='Lululemon'])}")
print(f"  ALO Yoga:  {len(tp[tp['brand']=='ALO Yoga'])}")
print(f"\nNulos restantes:")
print(tp.isnull().sum()[tp.isnull().sum() > 0])
tp.head()
```

- [ ] **Step 4: Celda 3 - Cargar Instagram publicaciones**

```markdown
## 2. Instagram - Publicaciones
```

```python
ig = pd.read_csv(f"{DATA_PROC}/instagram/ig_lululemon_publicaciones.csv")

# Limpieza
ig["caption_clean"] = ig["caption_std"].apply(clean_text)
ig["timestamp"] = pd.to_datetime(ig["timestamp_std"], errors="coerce")
ig["year"] = ig["timestamp"].dt.year
ig["month"] = ig["timestamp"].dt.to_period("M").astype(str)

# Renombrar columnas a esquema limpio
ig = ig.rename(columns={
    "likes_std": "likes",
    "comments_std": "comments",
    "url_std": "url",
})

# Engagement rate (sin views para posts, score ponderado)
ig = add_engagement_rate(ig, "likes", "comments")

# Eliminar duplicados por URL
n_before = len(ig)
ig = ig.drop_duplicates(subset="url", keep="first")
print(f"Instagram publicaciones: {n_before} -> {len(ig)} (eliminados {n_before - len(ig)} duplicados)")
print(f"Nulos restantes:")
print(ig.isnull().sum()[ig.isnull().sum() > 0])
ig.head()
```

- [ ] **Step 5: Celda 4 - Cargar Instagram comentarios**

```markdown
## 3. Instagram - Comentarios
```

```python
ig_com_us = pd.read_csv(f"{DATA_PROC}/instagram/ig_lululemon_comentarios_US.csv")
ig_com_eu = pd.read_csv(f"{DATA_PROC}/instagram/ig_lululemon_comentarios_EUROPE.csv")

ig_com = pd.concat([ig_com_us, ig_com_eu], ignore_index=True)
ig_com["platform"] = "instagram"
ig_com["text_clean"] = ig_com["text"].apply(clean_text)
ig_com["timestamp"] = pd.to_datetime(ig_com["timestamp"], errors="coerce", utc=True).dt.tz_localize(None)

# Eliminar comentarios vacios
n_before = len(ig_com)
ig_com = ig_com[ig_com["text_clean"].str.len() > 2].reset_index(drop=True)
print(f"Instagram comentarios: {n_before} -> {len(ig_com)} (eliminados {n_before - len(ig_com)} vacios)")

# Eliminar duplicados
ig_com = ig_com.drop_duplicates(subset=["comment_id"], keep="first")
print(f"Tras deduplicar: {len(ig_com)}")
ig_com.head()
```

- [ ] **Step 6: Celda 5 - Cargar TikTok videos**

```markdown
## 4. TikTok - Videos
```

```python
tk_us = pd.read_csv(f"{DATA_PROC}/tiktok/tiktok_lululemon_videos_US.csv")
tk_eu = pd.read_csv(f"{DATA_PROC}/tiktok/tiktok_lululemon_videos_EUROPE.csv")

tk_us["region_search"] = "US"
tk_eu["region_search"] = "Europe"

tk = pd.concat([tk_us, tk_eu], ignore_index=True)

# Limpieza
tk["title_clean"] = tk["title"].apply(clean_text)
tk["uploaded_at"] = pd.to_datetime(tk["uploadedAtFormatted"], errors="coerce")
tk["year"] = tk["uploaded_at"].dt.year
tk["month"] = tk["uploaded_at"].dt.to_period("M").astype(str)

# Engagement rate con views
tk = add_engagement_rate(tk, "likes", "comments", "views")

# Deduplicar por id
n_before = len(tk)
tk = tk.drop_duplicates(subset="id", keep="first")
print(f"TikTok videos: {n_before} -> {len(tk)} (eliminados {n_before - len(tk)} duplicados)")
tk.head()
```

- [ ] **Step 7: Celda 6 - Cargar TikTok comentarios**

```markdown
## 5. TikTok - Comentarios
```

```python
tk_com_us = pd.read_csv(f"{DATA_PROC}/tiktok/tiktok_lululemon_comentarios_US.csv")
tk_com_eu = pd.read_csv(f"{DATA_PROC}/tiktok/tiktok_lululemon_comentarios_EUROPE.csv")

tk_com_us["region_search"] = "US"
tk_com_eu["region_search"] = "Europe"

tk_com = pd.concat([tk_com_us, tk_com_eu], ignore_index=True)
tk_com["platform"] = "tiktok"
tk_com["text_clean"] = tk_com["text"].apply(clean_text)
tk_com["timestamp"] = pd.to_datetime(tk_com["createTimeISO"], errors="coerce")

# Renombrar para esquema uniforme
tk_com = tk_com.rename(columns={
    "cid": "comment_id",
    "diggCount": "likes",
    "uniqueId": "username",
})

# Eliminar vacios
n_before = len(tk_com)
tk_com = tk_com[tk_com["text_clean"].str.len() > 2].reset_index(drop=True)
print(f"TikTok comentarios: {n_before} -> {len(tk_com)} (eliminados {n_before - len(tk_com)} vacios)")
tk_com.head()
```

- [ ] **Step 8: Celda 7 - Resumen y guardado**

```markdown
## 6. Resumen y guardado de datos limpios
```

```python
# Guardar
tp.to_csv(f"{DATA_CLEAN}/trustpilot_all.csv", index=False, encoding="utf-8-sig")
ig.to_csv(f"{DATA_CLEAN}/ig_publicaciones.csv", index=False, encoding="utf-8-sig")
ig_com.to_csv(f"{DATA_CLEAN}/ig_comentarios.csv", index=False, encoding="utf-8-sig")
tk.to_csv(f"{DATA_CLEAN}/tiktok_videos.csv", index=False, encoding="utf-8-sig")
tk_com.to_csv(f"{DATA_CLEAN}/tiktok_comentarios.csv", index=False, encoding="utf-8-sig")

# Resumen
resumen = pd.DataFrame({
    "Dataset": ["Trustpilot Reviews", "Instagram Publicaciones", "Instagram Comentarios",
                 "TikTok Videos", "TikTok Comentarios"],
    "Filas": [len(tp), len(ig), len(ig_com), len(tk), len(tk_com)],
    "Columnas": [tp.shape[1], ig.shape[1], ig_com.shape[1], tk.shape[1], tk_com.shape[1]],
    "Nulos (%)": [
        round(tp.isnull().mean().mean()*100, 1),
        round(ig.isnull().mean().mean()*100, 1),
        round(ig_com.isnull().mean().mean()*100, 1),
        round(tk.isnull().mean().mean()*100, 1),
        round(tk_com.isnull().mean().mean()*100, 1),
    ],
})

resumen.to_csv("../../outputs/tablas/resumen_datasets.csv", index=False)
print("Datos limpios guardados en datos/clean/")
resumen
```

- [ ] **Step 9: Commit**

```bash
git add notebooks/analisis/01_limpieza_datos.ipynb datos/clean/
git commit -m "feat: add data cleaning notebook with unified schema"
```

---

## Task 2: Notebook 02 - EDA (Analisis Exploratorio)

**Files:**
- Create: `notebooks/analisis/02_EDA.ipynb`
- Output: `outputs/figuras/eda_*.png`, `outputs/tablas/eda_*.csv`

EDA exhaustivo: distribuciones, engagement, tendencias temporales, analisis de contenido (hashtags, top creators), correlaciones.

- [ ] **Step 1: Celda 0 - Titulo**

```markdown
# 02. Analisis Exploratorio de Datos (EDA)

Exploracion exhaustiva de los datos de Instagram, TikTok y Trustpilot para Lululemon.

**Secciones:**
1. Vision general de los datos
2. Analisis de engagement por plataforma
3. Tendencias temporales
4. Analisis de contenido (hashtags, captions, creators)
5. Distribuciones geograficas
6. Correlaciones
```

- [ ] **Step 2: Celda 1 - Setup**

```python
import sys
sys.path.insert(0, "../../src")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from plotting import setup_style, save_fig, COLORS, PLATFORM_COLORS
setup_style()

DATA = "../../datos/clean"

tp = pd.read_csv(f"{DATA}/trustpilot_all.csv", parse_dates=["published_date"])
ig = pd.read_csv(f"{DATA}/ig_publicaciones.csv", parse_dates=["timestamp"])
ig_com = pd.read_csv(f"{DATA}/ig_comentarios.csv", parse_dates=["timestamp"])
tk = pd.read_csv(f"{DATA}/tiktok_videos.csv", parse_dates=["uploaded_at"])
tk_com = pd.read_csv(f"{DATA}/tiktok_comentarios.csv", parse_dates=["timestamp"])

print(f"Trustpilot: {len(tp)} | IG posts: {len(ig)} | IG comments: {len(ig_com)}")
print(f"TikTok videos: {len(tk)} | TikTok comments: {len(tk_com)}")
```

- [ ] **Step 3: Celda 2 - Vision general**

```markdown
## 1. Vision general
```

```python
# Descripcion estadistica por plataforma
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# IG engagement
ig[["likes", "comments"]].describe().T[["mean", "50%", "max"]].plot(
    kind="bar", ax=axes[0], color=[PLATFORM_COLORS["instagram"], "#FF69B4"]
)
axes[0].set_title("Instagram - Engagement")
axes[0].set_ylabel("Valor")

# TikTok engagement
tk[["views", "likes", "comments", "shares", "bookmarks"]].describe().T[["mean", "50%", "max"]].plot(
    kind="bar", ax=axes[1], color=["#333", "#666", "#999", "#BBB", "#DDD"]
)
axes[1].set_title("TikTok - Engagement")
axes[1].set_ylabel("Valor")

# Trustpilot ratings
tp[tp["brand"]=="Lululemon"]["rating"].value_counts().sort_index().plot(
    kind="bar", ax=axes[2], color=COLORS["lululemon_red"]
)
axes[2].set_title("Trustpilot Lululemon - Ratings")
axes[2].set_xlabel("Estrellas")

plt.tight_layout()
save_fig(fig, "eda_overview")
plt.show()
```

- [ ] **Step 4: Celda 3 - Distribuciones de engagement**

```markdown
## 2. Distribuciones de engagement
```

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# IG likes distribution
axes[0,0].hist(ig["likes"], bins=50, color=PLATFORM_COLORS["instagram"], alpha=0.7, edgecolor="white")
axes[0,0].set_title("Instagram: Distribucion de Likes")
axes[0,0].set_xlabel("Likes")
axes[0,0].axvline(ig["likes"].median(), color="black", linestyle="--", label=f"Mediana: {ig['likes'].median():,.0f}")
axes[0,0].legend()

# IG comments distribution
axes[0,1].hist(ig["comments"], bins=50, color=PLATFORM_COLORS["instagram"], alpha=0.7, edgecolor="white")
axes[0,1].set_title("Instagram: Distribucion de Comentarios")
axes[0,1].set_xlabel("Comentarios")
axes[0,1].axvline(ig["comments"].median(), color="black", linestyle="--", label=f"Mediana: {ig['comments'].median():,.0f}")
axes[0,1].legend()

# TikTok views distribution
axes[1,0].hist(tk["views"], bins=50, color=PLATFORM_COLORS["tiktok"], alpha=0.7, edgecolor="white")
axes[1,0].set_title("TikTok: Distribucion de Views")
axes[1,0].set_xlabel("Views")
axes[1,0].axvline(tk["views"].median(), color="red", linestyle="--", label=f"Mediana: {tk['views'].median():,.0f}")
axes[1,0].legend()

# TikTok engagement rate
axes[1,1].hist(tk["engagement_rate"].dropna(), bins=50, color=PLATFORM_COLORS["tiktok"], alpha=0.7, edgecolor="white")
axes[1,1].set_title("TikTok: Engagement Rate (%)")
axes[1,1].set_xlabel("Engagement Rate")
axes[1,1].axvline(tk["engagement_rate"].median(), color="red", linestyle="--", label=f"Mediana: {tk['engagement_rate'].median():.1f}%")
axes[1,1].legend()

plt.tight_layout()
save_fig(fig, "eda_distribuciones_engagement")
plt.show()
```

- [ ] **Step 5: Celda 4 - Tendencias temporales**

```markdown
## 3. Tendencias temporales
```

```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# IG por mes
ig_monthly = ig.groupby(ig["timestamp"].dt.to_period("M")).agg(
    n_posts=("url", "count"),
    avg_likes=("likes", "mean"),
    avg_comments=("comments", "mean"),
).reset_index()
ig_monthly["timestamp"] = ig_monthly["timestamp"].astype(str)

ax1 = axes[0]
ax1.bar(range(len(ig_monthly)), ig_monthly["n_posts"], color=PLATFORM_COLORS["instagram"], alpha=0.6)
ax1.set_title("Instagram: Publicaciones por mes")
ax1.set_xlabel("Mes")
ax1.set_ylabel("N publicaciones")
ax1.set_xticks(range(0, len(ig_monthly), max(1, len(ig_monthly)//8)))
ax1.set_xticklabels(ig_monthly["timestamp"].iloc[::max(1, len(ig_monthly)//8)], rotation=45, ha="right")

# TikTok por mes
tk_monthly = tk.groupby(tk["uploaded_at"].dt.to_period("M")).agg(
    n_videos=("id", "count"),
    avg_views=("views", "mean"),
).reset_index()
tk_monthly["uploaded_at"] = tk_monthly["uploaded_at"].astype(str)

ax2 = axes[1]
ax2.bar(range(len(tk_monthly)), tk_monthly["n_videos"], color=PLATFORM_COLORS["tiktok"], alpha=0.6)
ax2.set_title("TikTok: Videos por mes")
ax2.set_xlabel("Mes")
ax2.set_ylabel("N videos")
ax2.set_xticks(range(0, len(tk_monthly), max(1, len(tk_monthly)//8)))
ax2.set_xticklabels(tk_monthly["uploaded_at"].iloc[::max(1, len(tk_monthly)//8)], rotation=45, ha="right")

plt.tight_layout()
save_fig(fig, "eda_tendencias_temporales")
plt.show()
```

- [ ] **Step 6: Celda 5 - Evolucion engagement**

```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# IG engagement evolution
ax1 = axes[0]
ax1.plot(range(len(ig_monthly)), ig_monthly["avg_likes"], marker="o", color=PLATFORM_COLORS["instagram"], label="Avg Likes")
ax1_2 = ax1.twinx()
ax1_2.plot(range(len(ig_monthly)), ig_monthly["avg_comments"], marker="s", color="#FF69B4", label="Avg Comments")
ax1.set_title("Instagram: Evolucion del engagement medio")
ax1.set_xlabel("Mes")
ax1.set_ylabel("Avg Likes")
ax1_2.set_ylabel("Avg Comments")
ax1.legend(loc="upper left")
ax1_2.legend(loc="upper right")

# TikTok views evolution
ax2 = axes[1]
ax2.plot(range(len(tk_monthly)), tk_monthly["avg_views"], marker="o", color=PLATFORM_COLORS["tiktok"])
ax2.set_title("TikTok: Evolucion de views medios")
ax2.set_xlabel("Mes")
ax2.set_ylabel("Avg Views")

plt.tight_layout()
save_fig(fig, "eda_evolucion_engagement")
plt.show()
```

- [ ] **Step 7: Celda 6 - Analisis de hashtags**

```markdown
## 4. Analisis de contenido
```

```python
import ast
from wordcloud import WordCloud

# Extraer hashtags de IG
def extract_hashtags(col):
    all_tags = []
    for val in col.dropna():
        try:
            tags = ast.literal_eval(val) if isinstance(val, str) else val
            if isinstance(tags, list):
                all_tags.extend([str(t).lower().strip() for t in tags])
        except:
            pass
    return Counter(all_tags)

ig_hashtags = extract_hashtags(ig["hashtags"])

# Top 20 hashtags IG
top_ig_tags = pd.DataFrame(ig_hashtags.most_common(20), columns=["hashtag", "count"])

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Barplot
axes[0].barh(top_ig_tags["hashtag"][::-1], top_ig_tags["count"][::-1], color=PLATFORM_COLORS["instagram"])
axes[0].set_title("Instagram: Top 20 Hashtags")
axes[0].set_xlabel("Frecuencia")

# Wordcloud
wc = WordCloud(width=800, height=400, background_color="white",
               colormap="RdPink" if "RdPink" in plt.colormaps() else "Reds").generate_from_frequencies(ig_hashtags)
axes[1].imshow(wc, interpolation="bilinear")
axes[1].axis("off")
axes[1].set_title("Instagram: Nube de Hashtags")

plt.tight_layout()
save_fig(fig, "eda_hashtags_instagram")
plt.show()
```

- [ ] **Step 8: Celda 7 - Top creators**

```python
# Top creators por engagement
top_ig_creators = ig.groupby("ownerUsername").agg(
    n_posts=("url", "count"),
    total_likes=("likes", "sum"),
    total_comments=("comments", "sum"),
    avg_likes=("likes", "mean"),
).sort_values("total_likes", ascending=False).head(15)

top_tk_creators = tk.groupby("author").agg(
    n_videos=("id", "count"),
    total_views=("views", "sum"),
    total_likes=("likes", "sum"),
    avg_views=("views", "mean"),
).sort_values("total_views", ascending=False).head(15)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

axes[0].barh(top_ig_creators.index[::-1], top_ig_creators["total_likes"][::-1], color=PLATFORM_COLORS["instagram"])
axes[0].set_title("Instagram: Top 15 Creators por Likes totales")
axes[0].set_xlabel("Total Likes")

axes[1].barh(top_tk_creators.index[::-1], top_tk_creators["total_views"][::-1], color=PLATFORM_COLORS["tiktok"])
axes[1].set_title("TikTok: Top 15 Creators por Views totales")
axes[1].set_xlabel("Total Views")

plt.tight_layout()
save_fig(fig, "eda_top_creators")
plt.show()
```

- [ ] **Step 9: Celda 8 - Distribucion geografica (Trustpilot)**

```markdown
## 5. Distribucion geografica
```

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Trustpilot Lululemon por pais
tp_lulu = tp[tp["brand"] == "Lululemon"]
country_counts = tp_lulu["country"].value_counts().head(10)
axes[0].barh(country_counts.index[::-1], country_counts.values[::-1], color=COLORS["lululemon_red"])
axes[0].set_title("Trustpilot Lululemon: Reviews por pais")
axes[0].set_xlabel("N reviews")

# Idiomas de comentarios combinados
all_comments = pd.concat([
    ig_com[["lang", "region"]].assign(platform="instagram"),
    tk_com[["region_search"]].rename(columns={"region_search": "region"}).assign(platform="tiktok", lang=""),
])
lang_counts = ig_com["lang"].value_counts().head(8)
axes[1].barh(lang_counts.index[::-1], lang_counts.values[::-1], color=COLORS["europe"])
axes[1].set_title("Instagram: Comentarios por idioma detectado")
axes[1].set_xlabel("N comentarios")

plt.tight_layout()
save_fig(fig, "eda_distribucion_geografica")
plt.show()
```

- [ ] **Step 10: Celda 9 - Correlaciones**

```markdown
## 6. Correlaciones
```

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# IG: likes vs comments
axes[0].scatter(ig["likes"], ig["comments"], alpha=0.4, s=20, color=PLATFORM_COLORS["instagram"])
axes[0].set_xlabel("Likes")
axes[0].set_ylabel("Comentarios")
axes[0].set_title(f"Instagram: Likes vs Comentarios (r={ig['likes'].corr(ig['comments']):.2f})")

# TikTok: views vs likes
axes[1].scatter(tk["views"], tk["likes"], alpha=0.4, s=20, color=PLATFORM_COLORS["tiktok"])
axes[1].set_xlabel("Views")
axes[1].set_ylabel("Likes")
axes[1].set_title(f"TikTok: Views vs Likes (r={tk['views'].corr(tk['likes']):.2f})")

plt.tight_layout()
save_fig(fig, "eda_correlaciones")
plt.show()
```

- [ ] **Step 11: Celda 10 - Heatmap de correlaciones TikTok**

```python
# Heatmap metricas TikTok
tk_metrics = tk[["views", "likes", "comments", "shares", "bookmarks"]].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(tk_metrics, annot=True, fmt=".2f", cmap="RdYlGn", center=0, ax=ax)
ax.set_title("TikTok: Correlacion entre metricas de engagement")
plt.tight_layout()
save_fig(fig, "eda_heatmap_tiktok")
plt.show()
```

- [ ] **Step 12: Celda 11 - Tipo de contenido IG**

```python
# Reels vs Posts en IG
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

kind_stats = ig.groupby("kind").agg(
    count=("url", "count"),
    avg_likes=("likes", "mean"),
    avg_comments=("comments", "mean"),
)

kind_stats["count"].plot(kind="bar", ax=axes[0], color=[PLATFORM_COLORS["instagram"], "#FF69B4"])
axes[0].set_title("Instagram: Posts vs Reels (cantidad)")
axes[0].set_ylabel("N publicaciones")
axes[0].tick_params(axis="x", rotation=0)

kind_stats[["avg_likes", "avg_comments"]].plot(kind="bar", ax=axes[1])
axes[1].set_title("Instagram: Engagement medio por tipo")
axes[1].set_ylabel("Promedio")
axes[1].tick_params(axis="x", rotation=0)

plt.tight_layout()
save_fig(fig, "eda_tipo_contenido_ig")
plt.show()
```

- [ ] **Step 13: Celda 12 - Resumen estadistico**

```python
# Exportar tabla resumen
resumen_engagement = pd.DataFrame({
    "Metrica": ["IG Likes (media)", "IG Likes (mediana)", "IG Comments (media)",
                 "TK Views (media)", "TK Views (mediana)", "TK Likes (media)",
                 "TK Engagement Rate (media)", "TP Rating Lulu (media)", "TP Rating ALO (media)"],
    "Valor": [
        f"{ig['likes'].mean():,.0f}", f"{ig['likes'].median():,.0f}", f"{ig['comments'].mean():,.0f}",
        f"{tk['views'].mean():,.0f}", f"{tk['views'].median():,.0f}", f"{tk['likes'].mean():,.0f}",
        f"{tk['engagement_rate'].mean():.1f}%",
        f"{tp[tp['brand']=='Lululemon']['rating'].mean():.2f}",
        f"{tp[tp['brand']=='ALO Yoga']['rating'].mean():.2f}",
    ]
})

resumen_engagement.to_csv("../../outputs/tablas/eda_resumen_engagement.csv", index=False)
resumen_engagement
```

- [ ] **Step 14: Commit**

```bash
git add notebooks/analisis/02_EDA.ipynb outputs/
git commit -m "feat: add comprehensive EDA notebook with engagement, temporal, and content analysis"
```

---

## Task 3: Notebook 03 - Analisis de Sentimiento

**Files:**
- Create: `notebooks/analisis/03_analisis_sentimiento.ipynb`
- Output: `datos/clean/*_sentiment.csv`, `outputs/figuras/sent_*.png`, `outputs/tablas/sent_*.csv`

BERT multilingual sobre todos los textos: comentarios IG, comentarios TK, reviews Trustpilot.

- [ ] **Step 1: Celda 0 - Titulo**

```markdown
# 03. Analisis de Sentimiento

Analisis de sentimiento multilingual con `nlptown/bert-base-multilingual-uncased-sentiment` (BERT fine-tuned para clasificar textos en 1-5 estrellas).

**Datos analizados:**
- Comentarios Instagram (US + Europa)
- Comentarios TikTok (US + Europa)
- Reviews Trustpilot (Lululemon + ALO Yoga)

**Clasificacion:**
- 1-2 estrellas → Negativo
- 3 estrellas → Neutro
- 4-5 estrellas → Positivo
```

- [ ] **Step 2: Celda 1 - Setup**

```python
import sys
sys.path.insert(0, "../../src")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentiment import predict_sentiment
from plotting import setup_style, save_fig, COLORS, PLATFORM_COLORS
setup_style()

DATA = "../../datos/clean"
```

- [ ] **Step 3: Celda 2 - Cargar datos**

```python
ig_com = pd.read_csv(f"{DATA}/ig_comentarios.csv")
tk_com = pd.read_csv(f"{DATA}/tiktok_comentarios.csv")
tp = pd.read_csv(f"{DATA}/trustpilot_all.csv")

print(f"IG comentarios: {len(ig_com)}")
print(f"TK comentarios: {len(tk_com)}")
print(f"Trustpilot reviews: {len(tp)}")
```

- [ ] **Step 4: Celda 3 - Sentimiento Instagram**

```markdown
## 1. Sentimiento - Comentarios Instagram
```

```python
print("Analizando sentimiento Instagram...")
ig_sent = predict_sentiment(ig_com["text_clean"].tolist())
ig_com = pd.concat([ig_com, ig_sent], axis=1)
print(f"Completado. Distribucion:")
print(ig_com["sentiment_label"].value_counts())
```

- [ ] **Step 5: Celda 4 - Sentimiento TikTok**

```markdown
## 2. Sentimiento - Comentarios TikTok
```

```python
print("Analizando sentimiento TikTok...")
tk_sent = predict_sentiment(tk_com["text_clean"].tolist())
tk_com = pd.concat([tk_com, tk_sent], axis=1)
print(f"Completado. Distribucion:")
print(tk_com["sentiment_label"].value_counts())
```

- [ ] **Step 6: Celda 5 - Sentimiento Trustpilot**

```markdown
## 3. Sentimiento - Reviews Trustpilot

Trustpilot ya tiene rating explicito (1-5), pero analizamos el sentimiento del texto para comparar.
```

```python
print("Analizando sentimiento Trustpilot...")
tp_sent = predict_sentiment(tp["text_clean"].tolist())
tp = pd.concat([tp, tp_sent], axis=1)

# Comparar rating real vs sentimiento predicho
tp["rating_label"] = pd.cut(tp["rating"], bins=[0, 2, 3, 5], labels=["negativo", "neutro", "positivo"])
concordancia = (tp["rating_label"] == tp["sentiment_label"]).mean()
print(f"\nConcordancia rating vs sentimiento BERT: {concordancia:.1%}")
print(f"\nDistribucion sentimiento BERT:")
print(tp["sentiment_label"].value_counts())
```

- [ ] **Step 7: Celda 6 - Visualizacion comparativa**

```markdown
## 4. Visualizacion del sentimiento por plataforma
```

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

sent_order = ["positivo", "neutro", "negativo"]
colors_sent = [COLORS["positive"], COLORS["neutral"], COLORS["negative"]]

# Instagram
ig_counts = ig_com["sentiment_label"].value_counts().reindex(sent_order)
axes[0].bar(sent_order, ig_counts.values, color=colors_sent)
axes[0].set_title(f"Instagram ({len(ig_com)} comentarios)")
for i, v in enumerate(ig_counts.values):
    axes[0].text(i, v + 5, f"{v}\n({v/len(ig_com)*100:.0f}%)", ha="center", fontsize=10)

# TikTok
tk_counts = tk_com["sentiment_label"].value_counts().reindex(sent_order)
axes[1].bar(sent_order, tk_counts.values, color=colors_sent)
axes[1].set_title(f"TikTok ({len(tk_com)} comentarios)")
for i, v in enumerate(tk_counts.values):
    axes[1].text(i, v + 5, f"{v}\n({v/len(tk_com)*100:.0f}%)", ha="center", fontsize=10)

# Trustpilot
tp_lulu = tp[tp["brand"] == "Lululemon"]
tp_counts = tp_lulu["sentiment_label"].value_counts().reindex(sent_order)
axes[2].bar(sent_order, tp_counts.values, color=colors_sent)
axes[2].set_title(f"Trustpilot Lululemon ({len(tp_lulu)} reviews)")
for i, v in enumerate(tp_counts.values):
    axes[2].text(i, v + 2, f"{v}\n({v/len(tp_lulu)*100:.0f}%)", ha="center", fontsize=10)

plt.suptitle("Distribucion del sentimiento por plataforma", fontsize=14, fontweight="bold")
plt.tight_layout()
save_fig(fig, "sent_por_plataforma")
plt.show()
```

- [ ] **Step 8: Celda 7 - Sentimiento por estrellas (detalle)**

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (label, df) in zip(axes, [("Instagram", ig_com), ("TikTok", tk_com), ("Trustpilot", tp_lulu)]):
    stars_dist = df["sentiment_stars"].value_counts().sort_index()
    ax.bar(stars_dist.index, stars_dist.values, color=[COLORS["negative"], COLORS["negative"],
           COLORS["neutral"], COLORS["positive"], COLORS["positive"]])
    ax.set_title(f"{label}: Distribucion por estrellas BERT")
    ax.set_xlabel("Estrellas (1-5)")
    ax.set_ylabel("N textos")
    ax.set_xticks([1, 2, 3, 4, 5])

plt.tight_layout()
save_fig(fig, "sent_estrellas_detalle")
plt.show()
```

- [ ] **Step 9: Celda 8 - Temas en sentimiento negativo**

```markdown
## 5. Que dicen los comentarios negativos?
```

```python
from wordcloud import WordCloud
from collections import Counter
import re

def get_top_words(texts, n=30, min_len=4):
    """Extrae las palabras mas frecuentes de una lista de textos."""
    stopwords = {"this", "that", "with", "have", "from", "they", "them", "their", "your", "would",
                 "been", "were", "will", "just", "more", "como", "para", "pero", "este", "esta",
                 "very", "much", "also", "about", "what", "when", "than", "some", "lululemon", "the"}
    words = []
    for text in texts:
        if pd.isna(text):
            continue
        tokens = re.findall(r'\b[a-zA-Záéíóúñü]+\b', str(text).lower())
        words.extend([w for w in tokens if len(w) >= min_len and w not in stopwords])
    return Counter(words).most_common(n)

# Comentarios negativos por plataforma
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (label, df) in zip(axes, [("Instagram", ig_com), ("TikTok", tk_com), ("Trustpilot", tp_lulu)]):
    neg_texts = df[df["sentiment_label"] == "negativo"]["text_clean"]
    if len(neg_texts) > 5:
        wc_data = dict(get_top_words(neg_texts))
        wc = WordCloud(width=600, height=300, background_color="white",
                       colormap="Reds").generate_from_frequencies(wc_data)
        ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"{label}: Temas negativos ({len(neg_texts)} textos)")

plt.tight_layout()
save_fig(fig, "sent_wordcloud_negativos")
plt.show()
```

- [ ] **Step 10: Celda 9 - Temas en sentimiento positivo**

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (label, df) in zip(axes, [("Instagram", ig_com), ("TikTok", tk_com), ("Trustpilot", tp_lulu)]):
    pos_texts = df[df["sentiment_label"] == "positivo"]["text_clean"]
    if len(pos_texts) > 5:
        wc_data = dict(get_top_words(pos_texts))
        wc = WordCloud(width=600, height=300, background_color="white",
                       colormap="Greens").generate_from_frequencies(wc_data)
        ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"{label}: Temas positivos ({len(pos_texts)} textos)")

plt.tight_layout()
save_fig(fig, "sent_wordcloud_positivos")
plt.show()
```

- [ ] **Step 11: Celda 10 - Topic Modeling con LDA (preprocesamiento)**

```markdown
## 6. Topic Modeling (LDA)

Extraccion automatica de topicos latentes en los comentarios y reviews usando Latent Dirichlet Allocation.
Se analizan los textos de las 3 plataformas combinados, y luego se cruzan topicos con sentimiento y region.
```

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re

# Combinar todos los textos con metadatos
all_texts = pd.concat([
    ig_com[["text_clean", "sentiment_label", "region"]].assign(platform="instagram"),
    tk_com[["text_clean", "sentiment_label", "region_search"]].rename(columns={"region_search": "region"}).assign(platform="tiktok"),
    tp[tp["brand"]=="Lululemon"][["text_clean", "sentiment_label", "country"]].rename(columns={"country": "region"}).assign(platform="trustpilot"),
], ignore_index=True)

# Filtrar textos muy cortos
all_texts = all_texts[all_texts["text_clean"].str.len() >= 15].reset_index(drop=True)
print(f"Textos para LDA: {len(all_texts)}")

# Stopwords multilingue (en + es + fr + de + marcas)
STOPWORDS_CUSTOM = [
    # English
    "the", "and", "for", "that", "this", "with", "you", "your", "are", "was", "have", "has",
    "from", "they", "them", "their", "would", "been", "were", "will", "just", "more", "also",
    "about", "what", "when", "than", "some", "like", "really", "very", "much", "know", "think",
    "love", "want", "need", "get", "got", "one", "can", "don", "not", "but", "all", "out",
    # Spanish
    "que", "los", "las", "del", "una", "con", "por", "para", "como", "pero", "mas", "sus",
    "este", "esta", "estos", "estas", "todo", "muy", "bien", "hay", "ser", "tiene", "desde",
    # French
    "les", "des", "une", "que", "pas", "pour", "dans", "sur", "avec", "est", "sont", "plus",
    "tres", "cette", "ces", "tout", "bien", "fait", "aussi", "comme",
    # German
    "und", "die", "der", "das", "ist", "nicht", "ein", "eine", "den", "dem", "auf", "mit",
    "ich", "sie", "sich", "von", "auch", "noch", "aber", "hat", "nur", "sehr",
    # Brand/noise
    "lululemon", "lulu", "yoga", "http", "https", "www", "com",
]

# Vectorizar
vectorizer = CountVectorizer(
    max_df=0.85,
    min_df=5,
    max_features=2000,
    stop_words=STOPWORDS_CUSTOM,
    token_pattern=r'\b[a-zA-Záéíóúñüàèìòùâêîôûäëïöü]{3,}\b',
)

dtm = vectorizer.fit_transform(all_texts["text_clean"])
feature_names = vectorizer.get_feature_names_out()
print(f"Vocabulario: {len(feature_names)} terminos")
print(f"Matriz doc-term: {dtm.shape}")
```

- [ ] **Step 12: Celda 11 - Entrenar LDA y visualizar topicos**

```python
# Entrenar LDA con 8 topicos
N_TOPICS = 8

lda = LatentDirichletAllocation(
    n_components=N_TOPICS,
    random_state=42,
    max_iter=20,
    learning_method="online",
    n_jobs=-1,
)

lda_output = lda.fit_transform(dtm)

# Mostrar topicos
def print_topics(model, feature_names, n_top_words=12):
    topics = {}
    for idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics[f"Topico {idx+1}"] = top_words
        print(f"Topico {idx+1}: {', '.join(top_words)}")
    return topics

print(f"=== {N_TOPICS} Topicos LDA ===\n")
topics_dict = print_topics(lda, feature_names)
```

- [ ] **Step 13: Celda 12 - Visualizar topicos con barplots**

```python
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for idx, ax in enumerate(axes.flat):
    if idx >= N_TOPICS:
        ax.axis("off")
        continue
    top_indices = lda.components_[idx].argsort()[:-11:-1]
    top_words = [feature_names[i] for i in top_indices]
    top_weights = [lda.components_[idx][i] for i in top_indices]

    ax.barh(top_words[::-1], top_weights[::-1], color=plt.cm.Set3(idx / N_TOPICS))
    ax.set_title(f"Topico {idx+1}", fontsize=11, fontweight="bold")
    ax.tick_params(axis="y", labelsize=9)

plt.suptitle("Topicos LDA: Palabras mas relevantes por topico", fontsize=14, fontweight="bold")
plt.tight_layout()
save_fig(fig, "sent_lda_topicos")
plt.show()
```

- [ ] **Step 14: Celda 13 - Asignar topico dominante a cada texto**

```python
# Topico dominante por documento
all_texts["topico_dominante"] = lda_output.argmax(axis=1) + 1
all_texts["topico_prob"] = lda_output.max(axis=1)

# Distribucion de topicos
print("Distribucion de topicos:")
print(all_texts["topico_dominante"].value_counts().sort_index())
```

- [ ] **Step 15: Celda 14 - Cruce topicos x sentimiento**

```markdown
### 6.1 Topicos por sentimiento
```

```python
# Heatmap topico x sentimiento
cross = pd.crosstab(all_texts["topico_dominante"], all_texts["sentiment_label"], normalize="index") * 100

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(cross[["positivo", "neutro", "negativo"]], annot=True, fmt=".1f", cmap="RdYlGn",
            ax=ax, cbar_kws={"label": "% textos"})
ax.set_title("Distribucion de sentimiento por topico (%)")
ax.set_ylabel("Topico")
ax.set_xlabel("Sentimiento")
plt.tight_layout()
save_fig(fig, "sent_lda_topicos_x_sentimiento")
plt.show()

# Tabla: topicos mas negativos
topic_sentiment = all_texts.groupby("topico_dominante").agg(
    n=("text_clean", "count"),
    sent_media=("sentiment_stars", "mean") if "sentiment_stars" in all_texts.columns else ("topico_prob", "mean"),
    pct_negativo=("sentiment_label", lambda x: (x == "negativo").mean() * 100),
    pct_positivo=("sentiment_label", lambda x: (x == "positivo").mean() * 100),
).round(1)

# Añadir etiqueta con top words
topic_sentiment["top_words"] = topic_sentiment.index.map(
    lambda t: ", ".join(topics_dict.get(f"Topico {t}", [])[:5])
)

print("\nTopicos ordenados por % negativo:")
display(topic_sentiment.sort_values("pct_negativo", ascending=False))
```

- [ ] **Step 16: Celda 15 - Cruce topicos x region**

```markdown
### 6.2 Topicos por region (US vs Europa)
```

```python
# Simplificar regiones
def simplify_region(r):
    if r in ["US", "US/GB"]:
        return "US"
    elif r in ["Europe", "ES", "FR", "GB", "DE"]:
        return "Europa"
    return "Otro"

all_texts["region_simple"] = all_texts["region"].apply(simplify_region)

# Heatmap topico x region
cross_region = pd.crosstab(all_texts["topico_dominante"],
                            all_texts["region_simple"], normalize="columns") * 100

fig, ax = plt.subplots(figsize=(8, 6))
cross_region_plot = cross_region[["US", "Europa"]].copy() if "Europa" in cross_region.columns else cross_region
sns.heatmap(cross_region_plot, annot=True, fmt=".1f", cmap="Blues", ax=ax,
            cbar_kws={"label": "% textos por region"})
ax.set_title("Que topicos dominan en cada region? (%)")
ax.set_ylabel("Topico")
ax.set_xlabel("Region")
plt.tight_layout()
save_fig(fig, "sent_lda_topicos_x_region")
plt.show()

# Diferencia US - Europa por topico
if "US" in cross_region.columns and "Europa" in cross_region.columns:
    diff = (cross_region["Europa"] - cross_region["US"]).sort_values()
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [COLORS["europe"] if v > 0 else COLORS["us"] for v in diff.values]
    ax.barh([f"Topico {i}" for i in diff.index], diff.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Diferencia (Europa - US) en % de textos")
    ax.set_title("Topicos mas presentes en Europa vs US")
    plt.tight_layout()
    save_fig(fig, "sent_lda_diferencia_eu_vs_us")
    plt.show()
```

- [ ] **Step 17: Celda 16 - Cruce topicos x plataforma**

```markdown
### 6.3 Topicos por plataforma
```

```python
cross_plat = pd.crosstab(all_texts["topico_dominante"],
                          all_texts["platform"], normalize="columns") * 100

fig, ax = plt.subplots(figsize=(10, 6))
cross_plat.plot(kind="bar", ax=ax, color=[PLATFORM_COLORS.get(p, "#999") for p in cross_plat.columns])
ax.set_title("Distribucion de topicos por plataforma")
ax.set_xlabel("Topico")
ax.set_ylabel("% textos")
ax.legend(title="Plataforma")
plt.tight_layout()
save_fig(fig, "sent_lda_topicos_x_plataforma")
plt.show()

# Guardar topicos
topics_df = pd.DataFrame(topics_dict)
topics_df.index = [f"Palabra {i+1}" for i in range(len(topics_df))]
topics_df.to_csv("../../outputs/tablas/sent_lda_topicos.csv")
topic_sentiment.to_csv("../../outputs/tablas/sent_lda_topicos_sentimiento.csv")
print("Topicos guardados en outputs/tablas/")
```

- [ ] **Step 18: Celda 17 - Guardar datos con sentimiento y topicos**

```markdown
## 7. Guardar datos con sentimiento y topicos
```

```python
# Guardar sentimiento
ig_com.to_csv(f"{DATA}/ig_comentarios_sentiment.csv", index=False, encoding="utf-8-sig")
tk_com.to_csv(f"{DATA}/tiktok_comentarios_sentiment.csv", index=False, encoding="utf-8-sig")
tp.to_csv(f"{DATA}/trustpilot_all_sentiment.csv", index=False, encoding="utf-8-sig")

# Guardar textos con topicos
all_texts.to_csv(f"{DATA}/all_texts_sentiment_topics.csv", index=False, encoding="utf-8-sig")

# Tabla resumen
resumen_sent = pd.DataFrame({
    "Plataforma": ["Instagram", "TikTok", "Trustpilot Lulu", "Trustpilot ALO"],
    "N textos": [len(ig_com), len(tk_com), len(tp_lulu), len(tp[tp["brand"]=="ALO Yoga"])],
    "% Positivo": [
        f"{(ig_com['sentiment_label']=='positivo').mean()*100:.1f}%",
        f"{(tk_com['sentiment_label']=='positivo').mean()*100:.1f}%",
        f"{(tp_lulu['sentiment_label']=='positivo').mean()*100:.1f}%",
        f"{(tp[tp['brand']=='ALO Yoga']['sentiment_label']=='positivo').mean()*100:.1f}%",
    ],
    "% Negativo": [
        f"{(ig_com['sentiment_label']=='negativo').mean()*100:.1f}%",
        f"{(tk_com['sentiment_label']=='negativo').mean()*100:.1f}%",
        f"{(tp_lulu['sentiment_label']=='negativo').mean()*100:.1f}%",
        f"{(tp[tp['brand']=='ALO Yoga']['sentiment_label']=='negativo').mean()*100:.1f}%",
    ],
    "Media estrellas": [
        f"{ig_com['sentiment_stars'].mean():.2f}",
        f"{tk_com['sentiment_stars'].mean():.2f}",
        f"{tp_lulu['sentiment_stars'].mean():.2f}",
        f"{tp[tp['brand']=='ALO Yoga']['sentiment_stars'].mean():.2f}",
    ],
})

resumen_sent.to_csv("../../outputs/tablas/sent_resumen.csv", index=False)
resumen_sent
```

- [ ] **Step 12: Commit**

```bash
git add notebooks/analisis/03_analisis_sentimiento.ipynb datos/clean/*_sentiment.csv outputs/
git commit -m "feat: add sentiment analysis with multilingual BERT across all platforms"
```

---

## Task 4: Notebook 04 - Comparativa US vs Europa

**Files:**
- Create: `notebooks/analisis/04_comparativa_us_vs_europa.ipynb`
- Output: `outputs/figuras/comp_*.png`, `outputs/tablas/comp_*.csv`

El notebook central para la estrategia: diferencias de engagement, sentimiento, contenido y percepcion entre US y Europa.

- [ ] **Step 1: Celda 0 - Titulo**

```markdown
# 04. Comparativa US vs Europa

Analisis central para la estrategia de expansion europea de Lululemon.

**Preguntas clave:**
1. Hay diferencias de engagement entre US y Europa?
2. El sentimiento hacia Lululemon es diferente en Europa?
3. Que contenido funciona mejor en cada region?
4. Cuales son los pain points especificos de Europa?
5. Que paises europeos tienen mayor potencial?
```

- [ ] **Step 2: Celda 1 - Setup**

```python
import sys
sys.path.insert(0, "../../src")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotting import setup_style, save_fig, COLORS
setup_style()

DATA = "../../datos/clean"

ig_com = pd.read_csv(f"{DATA}/ig_comentarios_sentiment.csv")
tk_com = pd.read_csv(f"{DATA}/tiktok_comentarios_sentiment.csv")
tp = pd.read_csv(f"{DATA}/trustpilot_all_sentiment.csv")
tk = pd.read_csv(f"{DATA}/tiktok_videos.csv")
```

- [ ] **Step 3: Celda 2 - Engagement US vs Europa (TikTok)**

```markdown
## 1. Engagement US vs Europa
```

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics = ["views", "likes", "comments", "shares"]
for ax, metric in zip(axes.flat, metrics):
    data_us = tk[tk["region_search"] == "US"][metric]
    data_eu = tk[tk["region_search"] == "Europe"][metric]

    bp = ax.boxplot([data_us, data_eu], labels=["US", "Europa"], patch_artist=True,
                    showfliers=False)
    bp["boxes"][0].set_facecolor(COLORS["us"])
    bp["boxes"][1].set_facecolor(COLORS["europe"])
    ax.set_title(f"TikTok: {metric.capitalize()}")
    ax.set_ylabel(metric.capitalize())

plt.suptitle("Engagement TikTok: US vs Europa", fontsize=14, fontweight="bold")
plt.tight_layout()
save_fig(fig, "comp_engagement_tiktok_us_vs_eu")
plt.show()

# Test estadistico
from scipy import stats
for metric in metrics:
    us_vals = tk[tk["region_search"]=="US"][metric].dropna()
    eu_vals = tk[tk["region_search"]=="Europe"][metric].dropna()
    stat, pval = stats.mannwhitneyu(us_vals, eu_vals, alternative="two-sided")
    print(f"{metric}: U={stat:.0f}, p={pval:.4f} {'***' if pval<0.01 else '**' if pval<0.05 else 'ns'}")
```

- [ ] **Step 4: Celda 3 - Sentimiento US vs Europa**

```markdown
## 2. Sentimiento US vs Europa
```

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sent_order = ["positivo", "neutro", "negativo"]
colors_sent = [COLORS["positive"], COLORS["neutral"], COLORS["negative"]]

# Instagram
for region, offset, color in [("US", -0.2, COLORS["us"]), ("GB", 0, "#9333EA"), ("ES", 0.2, COLORS["es"])]:
    sub = ig_com[ig_com["region"] == region]
    if len(sub) > 0:
        counts = sub["sentiment_label"].value_counts(normalize=True).reindex(sent_order, fill_value=0)
        axes[0].bar([x + offset for x in range(3)], counts.values, 0.2, label=region, color=color, alpha=0.8)

axes[0].set_xticks(range(3))
axes[0].set_xticklabels(sent_order)
axes[0].set_title("Instagram: Sentimiento por region")
axes[0].set_ylabel("Proporcion")
axes[0].legend()

# TikTok
for region, offset, color in [("US", -0.15, COLORS["us"]), ("Europe", 0.15, COLORS["europe"])]:
    sub = tk_com[tk_com["region_search"] == region]
    if len(sub) > 0:
        counts = sub["sentiment_label"].value_counts(normalize=True).reindex(sent_order, fill_value=0)
        axes[1].bar([x + offset for x in range(3)], counts.values, 0.3, label=region, color=color, alpha=0.8)

axes[1].set_xticks(range(3))
axes[1].set_xticklabels(sent_order)
axes[1].set_title("TikTok: Sentimiento por region")
axes[1].set_ylabel("Proporcion")
axes[1].legend()

plt.suptitle("Sentimiento US vs Europa", fontsize=14, fontweight="bold")
plt.tight_layout()
save_fig(fig, "comp_sentimiento_us_vs_eu")
plt.show()
```

- [ ] **Step 5: Celda 4 - Sentimiento media por region**

```python
# Tabla comparativa detallada
regions_ig = ig_com.groupby("region").agg(
    n=("text_clean", "count"),
    sent_media=("sentiment_stars", "mean"),
    pct_positivo=("sentiment_label", lambda x: (x == "positivo").mean() * 100),
    pct_negativo=("sentiment_label", lambda x: (x == "negativo").mean() * 100),
).round(1)

regions_tk = tk_com.groupby("region_search").agg(
    n=("text_clean", "count"),
    sent_media=("sentiment_stars", "mean"),
    pct_positivo=("sentiment_label", lambda x: (x == "positivo").mean() * 100),
    pct_negativo=("sentiment_label", lambda x: (x == "negativo").mean() * 100),
).round(1)

print("=== Instagram por region ===")
display(regions_ig)
print("\n=== TikTok por region ===")
display(regions_tk)

regions_ig.to_csv("../../outputs/tablas/comp_sentimiento_ig_por_region.csv")
regions_tk.to_csv("../../outputs/tablas/comp_sentimiento_tk_por_region.csv")
```

- [ ] **Step 6: Celda 5 - Pain points Europa (Trustpilot)**

```markdown
## 3. Pain points especificos de Europa

Analisis de los temas recurrentes en reviews negativos de Trustpilot para paises europeos.
```

```python
tp_lulu_eu = tp[(tp["brand"] == "Lululemon") & (tp["country"].isin(["ES", "FR", "GB", "DE", "IT", "NL", "BE"]))]
tp_lulu_eu_neg = tp_lulu_eu[tp_lulu_eu["sentiment_label"] == "negativo"]

print(f"Reviews europeas Lululemon: {len(tp_lulu_eu)}")
print(f"De las cuales negativas: {len(tp_lulu_eu_neg)} ({len(tp_lulu_eu_neg)/len(tp_lulu_eu)*100:.0f}%)")
print(f"\nPaises:")
print(tp_lulu_eu_neg["country"].value_counts())

# Mostrar reviews negativas mas representativas
print("\n--- Reviews negativas Europa (muestra) ---")
for _, row in tp_lulu_eu_neg.head(10).iterrows():
    print(f"\n[{row['country']}] Rating: {row['rating']}⭐ | {row['title_clean']}")
    print(f"  {row['text_clean'][:200]}...")
```

- [ ] **Step 7: Celda 6 - Rating por pais (Trustpilot)**

```python
# Rating medio por pais
tp_lulu = tp[tp["brand"] == "Lululemon"]
country_ratings = tp_lulu.groupby("country").agg(
    n=("rating", "count"),
    rating_medio=("rating", "mean"),
    pct_1_2=("rating", lambda x: (x <= 2).mean() * 100),
).sort_values("n", ascending=False)

country_ratings = country_ratings[country_ratings["n"] >= 3]  # minimo 3 reviews

fig, ax = plt.subplots(figsize=(10, 6))
colors = [COLORS["europe"] if c in ["ES", "FR", "GB", "DE", "IT", "NL", "BE"] else COLORS["us"]
          for c in country_ratings.index]
ax.barh(country_ratings.index[::-1], country_ratings["rating_medio"][::-1], color=colors[::-1])
ax.axvline(3, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("Rating medio")
ax.set_title("Trustpilot Lululemon: Rating medio por pais")

for i, (idx, row) in enumerate(country_ratings[::-1].iterrows()):
    ax.text(row["rating_medio"] + 0.05, i, f"n={int(row['n'])}", va="center", fontsize=9)

plt.tight_layout()
save_fig(fig, "comp_rating_por_pais")
plt.show()
```

- [ ] **Step 8: Celda 7 - Resumen comparativa**

```python
# Tabla resumen ejecutiva
resumen_comp = pd.DataFrame({
    "Metrica": [
        "Sentimiento medio IG (US)", "Sentimiento medio IG (Europa)",
        "Sentimiento medio TK (US)", "Sentimiento medio TK (Europa)",
        "Rating Trustpilot (media global)", "% Negativo Trustpilot Europa",
    ],
    "Valor": [
        f"{ig_com[ig_com['region']=='US']['sentiment_stars'].mean():.2f}",
        f"{ig_com[ig_com['region']!='US']['sentiment_stars'].mean():.2f}",
        f"{tk_com[tk_com['region_search']=='US']['sentiment_stars'].mean():.2f}",
        f"{tk_com[tk_com['region_search']=='Europe']['sentiment_stars'].mean():.2f}",
        f"{tp_lulu['rating'].mean():.2f}",
        f"{(tp_lulu_eu['sentiment_label']=='negativo').mean()*100:.1f}%",
    ]
})
resumen_comp.to_csv("../../outputs/tablas/comp_resumen_us_vs_eu.csv", index=False)
resumen_comp
```

- [ ] **Step 9: Commit**

```bash
git add notebooks/analisis/04_comparativa_us_vs_europa.ipynb outputs/
git commit -m "feat: add US vs Europe comparative analysis with engagement and sentiment"
```

---

## Task 5: Notebook 05 - Competencia Lululemon vs ALO Yoga

**Files:**
- Create: `notebooks/analisis/05_competencia_lululemon_vs_alo.ipynb`
- Output: `outputs/figuras/bench_*.png`, `outputs/tablas/bench_*.csv`

- [ ] **Step 1: Celda 0 - Titulo**

```markdown
# 05. Benchmarking: Lululemon vs ALO Yoga

Comparativa de la percepcion de marca entre Lululemon y ALO Yoga a partir de reviews de Trustpilot.

**Dimensiones:**
1. Rating general
2. Sentimiento del texto
3. Temas recurrentes
4. Diferencias por mercado (pais)
```

- [ ] **Step 2: Celda 1 - Setup y carga**

```python
import sys
sys.path.insert(0, "../../src")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
from plotting import setup_style, save_fig, COLORS
setup_style()

tp = pd.read_csv("../../datos/clean/trustpilot_all_sentiment.csv", parse_dates=["published_date"])
lulu = tp[tp["brand"] == "Lululemon"].copy()
alo = tp[tp["brand"] == "ALO Yoga"].copy()

print(f"Lululemon: {len(lulu)} reviews | ALO Yoga: {len(alo)} reviews")
```

- [ ] **Step 3: Celda 2 - Rating comparison**

```markdown
## 1. Comparativa de ratings
```

```python
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Distribucion de ratings
for ax, (brand, df, color) in zip(axes[:2], [("Lululemon", lulu, COLORS["lululemon_red"]),
                                               ("ALO Yoga", alo, COLORS["alo_green"])]):
    rating_dist = df["rating"].value_counts().sort_index()
    ax.bar(rating_dist.index, rating_dist.values, color=color, edgecolor="white")
    ax.set_title(f"{brand} (media: {df['rating'].mean():.2f})")
    ax.set_xlabel("Estrellas")
    ax.set_ylabel("N reviews")
    ax.set_xticks([1, 2, 3, 4, 5])

# Comparativa directa
ratings_comp = pd.DataFrame({
    "Lululemon": lulu["rating"].value_counts(normalize=True).sort_index(),
    "ALO Yoga": alo["rating"].value_counts(normalize=True).sort_index(),
})
ratings_comp.plot(kind="bar", ax=axes[2], color=[COLORS["lululemon_red"], COLORS["alo_green"]])
axes[2].set_title("Distribucion comparada (%)")
axes[2].set_xlabel("Estrellas")
axes[2].set_ylabel("Proporcion")
axes[2].tick_params(axis="x", rotation=0)

plt.suptitle("Rating Trustpilot: Lululemon vs ALO Yoga", fontsize=14, fontweight="bold")
plt.tight_layout()
save_fig(fig, "bench_ratings_comparativa")
plt.show()
```

- [ ] **Step 4: Celda 3 - Sentimiento comparado**

```markdown
## 2. Sentimiento del texto
```

```python
fig, ax = plt.subplots(figsize=(10, 6))

sent_order = ["positivo", "neutro", "negativo"]
x = np.arange(len(sent_order))
width = 0.35

lulu_pcts = lulu["sentiment_label"].value_counts(normalize=True).reindex(sent_order, fill_value=0)
alo_pcts = alo["sentiment_label"].value_counts(normalize=True).reindex(sent_order, fill_value=0)

bars1 = ax.bar(x - width/2, lulu_pcts.values * 100, width, label="Lululemon", color=COLORS["lululemon_red"])
bars2 = ax.bar(x + width/2, alo_pcts.values * 100, width, label="ALO Yoga", color=COLORS["alo_green"])

ax.set_xticks(x)
ax.set_xticklabels(sent_order)
ax.set_ylabel("Porcentaje (%)")
ax.set_title("Sentimiento BERT: Lululemon vs ALO Yoga")
ax.legend()
ax.bar_label(bars1, fmt="%.1f%%")
ax.bar_label(bars2, fmt="%.1f%%")

plt.tight_layout()
save_fig(fig, "bench_sentimiento_comparativa")
plt.show()
```

- [ ] **Step 5: Celda 4 - Wordclouds por marca**

```markdown
## 3. Temas recurrentes por marca
```

```python
def get_top_words(texts, n=40, min_len=4):
    stopwords = {"this", "that", "with", "have", "from", "they", "them", "their", "your", "would",
                 "been", "were", "will", "just", "more", "como", "para", "pero", "este", "esta",
                 "very", "much", "also", "about", "what", "when", "than", "some", "lululemon", "yoga"}
    words = []
    for text in texts:
        if pd.isna(text):
            continue
        tokens = re.findall(r'\b[a-zA-Záéíóúñü]+\b', str(text).lower())
        words.extend([w for w in tokens if len(w) >= min_len and w not in stopwords])
    return Counter(words).most_common(n)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for col, (brand, df, cmap) in enumerate([("Lululemon", lulu, "Reds"), ("ALO Yoga", alo, "Greens")]):
    # Negativos
    neg_words = dict(get_top_words(df[df["sentiment_label"]=="negativo"]["text_clean"]))
    if neg_words:
        wc = WordCloud(width=600, height=300, background_color="white", colormap=cmap).generate_from_frequencies(neg_words)
        axes[0, col].imshow(wc, interpolation="bilinear")
    axes[0, col].axis("off")
    axes[0, col].set_title(f"{brand}: Temas NEGATIVOS")

    # Positivos
    pos_words = dict(get_top_words(df[df["sentiment_label"]=="positivo"]["text_clean"]))
    if pos_words:
        wc = WordCloud(width=600, height=300, background_color="white", colormap=cmap).generate_from_frequencies(pos_words)
        axes[1, col].imshow(wc, interpolation="bilinear")
    axes[1, col].axis("off")
    axes[1, col].set_title(f"{brand}: Temas POSITIVOS")

plt.tight_layout()
save_fig(fig, "bench_wordclouds_marca")
plt.show()
```

- [ ] **Step 6: Celda 5 - Evolucion temporal**

```python
# Rating medio mensual
fig, ax = plt.subplots(figsize=(12, 5))

for brand, color in [("Lululemon", COLORS["lululemon_red"]), ("ALO Yoga", COLORS["alo_green"])]:
    sub = tp[tp["brand"] == brand].copy()
    monthly = sub.groupby(sub["published_date"].dt.to_period("M"))["rating"].mean()
    monthly.index = monthly.index.astype(str)
    ax.plot(range(len(monthly)), monthly.values, marker="o", label=brand, color=color)
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels(monthly.index, rotation=45, ha="right")

ax.set_title("Evolucion del rating medio mensual")
ax.set_ylabel("Rating medio")
ax.legend()
ax.axhline(3, color="gray", linestyle="--", alpha=0.4)
plt.tight_layout()
save_fig(fig, "bench_evolucion_rating")
plt.show()
```

- [ ] **Step 7: Celda 6 - Tabla resumen**

```python
bench_resumen = pd.DataFrame({
    "Metrica": ["N reviews", "Rating medio", "% Positivo (BERT)", "% Negativo (BERT)",
                 "Sent. medio (estrellas)", "Reviews verificadas (%)"],
    "Lululemon": [
        len(lulu), f"{lulu['rating'].mean():.2f}",
        f"{(lulu['sentiment_label']=='positivo').mean()*100:.1f}%",
        f"{(lulu['sentiment_label']=='negativo').mean()*100:.1f}%",
        f"{lulu['sentiment_stars'].mean():.2f}",
        f"{lulu['is_verified'].mean()*100:.1f}%",
    ],
    "ALO Yoga": [
        len(alo), f"{alo['rating'].mean():.2f}",
        f"{(alo['sentiment_label']=='positivo').mean()*100:.1f}%",
        f"{(alo['sentiment_label']=='negativo').mean()*100:.1f}%",
        f"{alo['sentiment_stars'].mean():.2f}",
        f"{alo['is_verified'].mean()*100:.1f}%",
    ],
})
bench_resumen.to_csv("../../outputs/tablas/bench_resumen.csv", index=False)
bench_resumen
```

- [ ] **Step 8: Commit**

```bash
git add notebooks/analisis/05_competencia_lululemon_vs_alo.ipynb outputs/
git commit -m "feat: add competitive benchmarking Lululemon vs ALO Yoga"
```

---

## Task 6: Notebook 06 - Insights y Estrategia

**Files:**
- Create: `notebooks/analisis/06_insights_estrategia.ipynb`
- Output: `outputs/figuras/strategy_*.png`, `outputs/tablas/strategy_*.csv`

Sintesis de todos los hallazgos, traducidos a recomendaciones estrategicas.

- [ ] **Step 1: Celda 0 - Titulo**

```markdown
# 06. Insights y Recomendaciones Estrategicas

Sintesis de hallazgos del analisis multi-plataforma para disenar la estrategia de marketing de Lululemon en Europa.

**Objetivo:** Disenar una propuesta de estrategia de marketing basada en datos que permita a Lululemon acelerar su crecimiento en Europa y reforzar su posicionamiento como marca premium de referencia dentro del activewear.
```

- [ ] **Step 2: Celda 1 - Setup**

```python
import sys
sys.path.insert(0, "../../src")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotting import setup_style, save_fig, COLORS
setup_style()

# Cargar tablas resumen
resumen_sent = pd.read_csv("../../outputs/tablas/sent_resumen.csv")
resumen_comp = pd.read_csv("../../outputs/tablas/comp_resumen_us_vs_eu.csv")
resumen_bench = pd.read_csv("../../outputs/tablas/bench_resumen.csv")
sent_ig_region = pd.read_csv("../../outputs/tablas/comp_sentimiento_ig_por_region.csv", index_col=0)
sent_tk_region = pd.read_csv("../../outputs/tablas/comp_sentimiento_tk_por_region.csv", index_col=0)

# Datos completos para analisis adicional
DATA = "../../datos/clean"
ig_com = pd.read_csv(f"{DATA}/ig_comentarios_sentiment.csv")
tk_com = pd.read_csv(f"{DATA}/tiktok_comentarios_sentiment.csv")
tp = pd.read_csv(f"{DATA}/trustpilot_all_sentiment.csv")
tk = pd.read_csv(f"{DATA}/tiktok_videos.csv")
```

- [ ] **Step 3: Celda 2 - Dashboard resumen**

```markdown
## 1. Dashboard: Estado actual de Lululemon
```

```python
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Sentimiento global por plataforma
sent_data = resumen_sent.set_index("Plataforma")
for i, plat in enumerate(["Instagram", "TikTok", "Trustpilot Lulu"]):
    val = float(sent_data.loc[plat, "% Positivo"].strip("%"))
    neg = float(sent_data.loc[plat, "% Negativo"].strip("%"))
    neu = 100 - val - neg
    axes[0,i].pie([val, neu, neg], labels=["Pos", "Neu", "Neg"],
                   colors=[COLORS["positive"], COLORS["neutral"], COLORS["negative"]],
                   autopct="%1.0f%%", startangle=90)
    axes[0,i].set_title(f"{plat}")

# 2. Gap US vs Europa
metrics_gap = {
    "Sent. IG": [
        ig_com[ig_com["region"]=="US"]["sentiment_stars"].mean(),
        ig_com[ig_com["region"]!="US"]["sentiment_stars"].mean(),
    ],
    "Sent. TK": [
        tk_com[tk_com["region_search"]=="US"]["sentiment_stars"].mean(),
        tk_com[tk_com["region_search"]=="Europe"]["sentiment_stars"].mean(),
    ],
}

x = np.arange(len(metrics_gap))
width = 0.35
ax = axes[1,0]
us_vals = [v[0] for v in metrics_gap.values()]
eu_vals = [v[1] for v in metrics_gap.values()]
ax.bar(x - width/2, us_vals, width, label="US", color=COLORS["us"])
ax.bar(x + width/2, eu_vals, width, label="Europa", color=COLORS["europe"])
ax.set_xticks(x)
ax.set_xticklabels(list(metrics_gap.keys()))
ax.set_title("Gap sentimiento US vs Europa")
ax.legend()
ax.set_ylabel("Media estrellas (1-5)")

# 3. Lululemon vs ALO
ax = axes[1,1]
brands = ["Lululemon", "ALO Yoga"]
ratings = [float(resumen_bench[resumen_bench.columns[1]].iloc[1]),
           float(resumen_bench[resumen_bench.columns[2]].iloc[1])]
ax.bar(brands, ratings, color=[COLORS["lululemon_red"], COLORS["alo_green"]])
ax.set_title("Rating Trustpilot medio")
ax.set_ylabel("Rating (1-5)")
ax.set_ylim(0, 5)

# 4. Engagement TikTok US vs EU
ax = axes[1,2]
eng_us = tk[tk["region_search"]=="US"]["engagement_rate"].median()
eng_eu = tk[tk["region_search"]=="Europe"]["engagement_rate"].median()
ax.bar(["US", "Europa"], [eng_us, eng_eu], color=[COLORS["us"], COLORS["europe"]])
ax.set_title("TikTok: Engagement Rate mediano")
ax.set_ylabel("Engagement Rate (%)")

plt.suptitle("DASHBOARD: Estado actual de Lululemon en redes sociales", fontsize=16, fontweight="bold")
plt.tight_layout()
save_fig(fig, "strategy_dashboard")
plt.show()
```

- [ ] **Step 4: Celda 3 - Hallazgos clave**

```markdown
## 2. Hallazgos Clave

### Hallazgo 1: Gap de sentimiento US vs Europa
El sentimiento hacia Lululemon es consistentemente inferior en Europa comparado con US en todas las plataformas analizadas.

### Hallazgo 2: Pain points europeos diferenciados
Los reviews negativos europeos se centran en logistica (envios, devoluciones) y atencion al cliente, mientras que en US son mas sobre producto.

### Hallazgo 3: Oportunidad en redes sociales europeas
El engagement en TikTok europeo muestra potencial de crecimiento, con menos saturacion que US.

### Hallazgo 4: Competencia (ALO Yoga)
ALO Yoga compite en el mismo segmento con una percepcion mixta - oportunidad de diferenciacion.
```

- [ ] **Step 5: Celda 4 - SWOT basado en datos**

```markdown
## 3. Analisis SWOT basado en datos
```

```python
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis("off")

swot = {
    "FORTALEZAS": [
        "Alto engagement en IG (media likes elevada)",
        "Sentimiento positivo dominante en RRSS",
        "Comunidad activa de creators (hashtags)",
        "Brand awareness fuerte en activewear",
    ],
    "DEBILIDADES": [
        "Logistica europea deficiente (envios, devoluciones)",
        "Atencion al cliente percibida como mala en Europa",
        "Rating Trustpilot bajo en mercados EU clave",
        "Menor presencia de contenido localizado",
    ],
    "OPORTUNIDADES": [
        "TikTok europeo con menor competencia",
        "Mercados ES, FR, DE con audiencia receptiva",
        "UGC y micro-influencers locales",
        "Gap respecto a ALO en experiencia post-venta",
    ],
    "AMENAZAS": [
        "Competidores con mejor logistica local (Zara, H&M Sport)",
        "Percepcion negativa puede viralizarse",
        "Diferencias culturales en marketing de fitness",
        "Sensibilidad al precio en mercados EU",
    ],
}

colors_swot = {"FORTALEZAS": "#22C55E", "DEBILIDADES": "#EF4444",
               "OPORTUNIDADES": "#3B82F6", "AMENAZAS": "#F59E0B"}

for i, (title, items) in enumerate(swot.items()):
    row, col = divmod(i, 2)
    x, y = 0.05 + col * 0.5, 0.95 - row * 0.5
    ax.text(x, y, title, fontsize=14, fontweight="bold", color=colors_swot[title],
            transform=ax.transAxes)
    for j, item in enumerate(items):
        ax.text(x + 0.02, y - 0.07 - j * 0.08, f"• {item}", fontsize=10,
                transform=ax.transAxes, wrap=True)

ax.set_title("SWOT Lululemon Europa - Basado en datos", fontsize=16, fontweight="bold", pad=20)
plt.tight_layout()
save_fig(fig, "strategy_swot")
plt.show()
```

- [ ] **Step 6: Celda 5 - Recomendaciones estrategicas**

```markdown
## 4. Recomendaciones Estrategicas

### R1: Mejorar la experiencia post-venta en Europa
**Evidencia:** X% de reviews negativas en Trustpilot Europa mencionan logistica y devoluciones.
**Accion:** Partnering con operadores logisticos locales, puntos de recogida, atencion al cliente en idioma local.

### R2: Estrategia de contenido localizado en TikTok
**Evidencia:** El engagement europeo en TikTok muestra potencial con menor saturacion.
**Accion:** Programa de micro-influencers locales por pais (ES, FR, DE, GB), contenido en idioma local.

### R3: Programa de UGC (User Generated Content) europeo
**Evidencia:** Los hashtags de Lululemon tienen comunidad activa pero centrada en US.
**Accion:** Campanas de hashtag localizadas, challenges TikTok por mercado.

### R4: Diferenciacion vs ALO Yoga
**Evidencia:** ALO Yoga tiene debilidades en atencion al cliente similares.
**Accion:** Posicionar la mejora de servicio como ventaja competitiva, no solo producto.

### R5: Priorizar mercados por potencial
**Evidencia:** Analisis de sentimiento y engagement por pais.
**Accion:** Foco en GB (mercado maduro), ES y FR (crecimiento), DE (oportunidad).
```

- [ ] **Step 7: Celda 6 - Tabla resumen final**

```python
# Tabla resumen de recomendaciones con KPIs
recomendaciones = pd.DataFrame({
    "Recomendacion": [
        "R1: Mejorar logistica EU",
        "R2: Contenido TikTok local",
        "R3: Programa UGC europeo",
        "R4: Diferenciacion vs ALO",
        "R5: Priorizacion por mercado",
    ],
    "Prioridad": ["Alta", "Alta", "Media", "Media", "Alta"],
    "KPI objetivo": [
        "Subir rating TP Europa a 3.5+",
        "Engagement rate TK EU > 5%",
        "500+ posts con hashtag local/mes",
        "NPS superior a ALO en 6 meses",
        "Presencia activa en GB, ES, FR",
    ],
    "Evidencia (notebook)": [
        "NB04: Pain points EU",
        "NB04: Gap engagement US/EU",
        "NB02: Analisis hashtags",
        "NB05: Benchmarking",
        "NB04: Sentimiento por pais",
    ],
})

recomendaciones.to_csv("../../outputs/tablas/strategy_recomendaciones.csv", index=False)
recomendaciones
```

- [ ] **Step 8: Commit**

```bash
git add notebooks/analisis/06_insights_estrategia.ipynb outputs/
git commit -m "feat: add strategic insights notebook with SWOT and recommendations"
```

---

## Task 7: Validacion final y limpieza

- [ ] **Step 1: Verificar que todos los notebooks ejecutan sin errores**

```bash
cd notebooks/analisis
for nb in 01_limpieza_datos.ipynb 02_EDA.ipynb 03_analisis_sentimiento.ipynb 04_comparativa_us_vs_europa.ipynb 05_competencia_lululemon_vs_alo.ipynb 06_insights_estrategia.ipynb; do
    echo "=== Ejecutando $nb ==="
    jupyter nbconvert --to notebook --execute "$nb" --ExecutePreprocessor.timeout=600 2>&1 | tail -1
done
```

- [ ] **Step 2: Verificar outputs generados**

```bash
echo "=== Figuras ==="
ls -la ../../outputs/figuras/
echo "=== Tablas ==="
ls -la ../../outputs/tablas/
echo "=== Datos clean ==="
ls -la ../../datos/clean/
```

- [ ] **Step 3: Commit final**

```bash
git add -A
git commit -m "feat: complete Lululemon Europe analytics pipeline - 6 analysis notebooks"
```
