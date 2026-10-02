"""Rutas canónicas del proyecto.

Todas las rutas se construyen a partir de la raíz del repositorio, de modo que
cualquier script pueda ejecutarse desde cualquier directorio de trabajo sin
depender de rutas relativas frágiles.
"""

from __future__ import annotations

from pathlib import Path

# common -> src -> raíz del proyecto
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# --- Capas del proyecto -------------------------------------------------
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"
DATA_PROCESSED_DIR: Path = DATA_DIR / "processed"

SRC_DIR: Path = PROJECT_ROOT / "src"
MODELS_DIR: Path = PROJECT_ROOT / "models"
VISUALIZATIONS_DIR: Path = PROJECT_ROOT / "visualizations"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
TESTS_DIR: Path = PROJECT_ROOT / "tests"
REPORT_DIR: Path = PROJECT_ROOT / "informe"


def ruta_cruda(nombre: str) -> Path:
    """Devuelve la ruta de un dataset crudo dentro de ``data/raw``."""
    return DATA_RAW_DIR / nombre


def ruta_procesada(nombre: str) -> Path:
    """Devuelve la ruta de un dataset procesado dentro de ``data/processed``."""
    return DATA_PROCESSED_DIR / nombre


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "DATA_RAW_DIR",
    "DATA_PROCESSED_DIR",
    "SRC_DIR",
    "MODELS_DIR",
    "VISUALIZATIONS_DIR",
    "RESULTS_DIR",
    "TESTS_DIR",
    "REPORT_DIR",
    "ruta_cruda",
    "ruta_procesada",
]
