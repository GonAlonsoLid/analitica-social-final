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
        df = df.copy()
        df["engagement_rate"] = ((df[likes_col] + df[comments_col]) / df[views_col].replace(0, np.nan)) * 100
    else:
        df = df.copy()
        df["engagement_rate"] = df[likes_col] + df[comments_col] * 3  # score ponderado
    return df
