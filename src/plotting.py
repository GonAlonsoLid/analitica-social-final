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
