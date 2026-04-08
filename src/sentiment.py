"""Analisis de sentimiento con BERT multilingual."""
import pandas as pd
import numpy as np
from transformers import pipeline
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
