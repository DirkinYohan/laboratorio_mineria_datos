"""Limpieza y preparación de los datos del modelo de nivel de glucosa.

Pipeline reproducible, sin intervención manual:

    cargar crudo -> auditar -> seleccionar columnas -> tipar
    -> eliminar filas inválidas -> separar X e y -> guardar en data/processed

El dataset original de ``data/raw`` nunca se modifica.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.common import io, limpieza
from src.common.limpieza import Auditoria
from .especificacion import ESPECIFICACION


def cargar_crudo() -> pd.DataFrame:
    """Carga el dataset crudo ``data/raw/glucosa_data.csv``."""
    return io.cargar_dataset(ESPECIFICACION.ruta_cruda)


def validar_esquema(df: pd.DataFrame) -> pd.DataFrame:
    """Verifica que el dataset contenga las columnas esperadas.

    Raises:
        ValueError: Si falta alguna variable independiente o la objetivo.
    """
    faltantes = [c for c in ESPECIFICACION.columnas if c not in df.columns]
    if faltantes:
        raise ValueError(
            f"Columnas faltantes en '{ESPECIFICACION.archivo_crudo}': {faltantes}. "
            f"Se esperaban: {list(ESPECIFICACION.columnas)}."
        )
    return df


def auditar(df: pd.DataFrame) -> Auditoria:
    """Genera el reporte de auditoría de calidad del dataset."""
    return limpieza.auditar(df, ESPECIFICACION)


def describir(df: pd.DataFrame) -> pd.DataFrame:
    """Estadística descriptiva de las variables del modelo."""
    return limpieza.describir(df, ESPECIFICACION.columnas)


def limpiar(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Limpia el dataset crudo y documenta lo que hizo.

    Operaciones aplicadas (comprobadas sobre los datos reales):

    1. Se conservan únicamente las cuatro variables del modelo.
    2. Se fijan los tipos: ``Edad`` y ``Actividad_Fisica`` enteros; ``IMC`` y
       ``Nivel_Glucosa`` en punto flotante.
    3. Se eliminan filas con nulos, duplicados exactos y valores imposibles.
    4. Los valores atípicos se detectan y se documentan, pero **no se
       eliminan**: representan casos clínicos reales (IMC elevado, glucosa alta)
       y su eliminación sesgaría el modelo.

    Args:
        df: Dataset crudo ya cargado.

    Returns:
        ``(dataset_limpio, reporte)`` con el detalle de la limpieza.
    """
    filas_antes = len(df)

    trabajo = limpieza.seleccionar_columnas(df, ESPECIFICACION)
    trabajo = limpieza.ajustar_tipos(trabajo, ESPECIFICACION)
    trabajo, eliminaciones = limpieza.eliminar_filas_invalidas(trabajo, ESPECIFICACION)

    reporte = limpieza.resumen_limpieza(
        ESPECIFICACION, filas_antes, eliminaciones, len(trabajo)
    )
    reporte["nota_codificacion"] = (
        "Se conservó 'Actividad_Fisica' como variable ordinal (0-9). Se comparó "
        "con su codificación en variables ficticias y la diferencia de R² fue "
        "despreciable, por lo que se mantiene la escala original, más "
        "interpretable y con menos parámetros."
    )
    return trabajo, reporte


def separar_variables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa la matriz de variables independientes y el vector objetivo.

    Returns:
        ``(X, y)`` con las columnas en el orden declarado.
    """
    X = df.loc[:, list(ESPECIFICACION.variables)].copy()
    y = df[ESPECIFICACION.objetivo].copy()
    return X, y


def preparar(*, persistir: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Ejecuta el pipeline completo de preparación.

    Args:
        persistir: Si es ``True``, guarda el dataset limpio en
            ``data/processed`` y el reporte en ``results/glucosa``.

    Returns:
        ``(X, y)`` listos para el entrenamiento.
    """
    crudo = cargar_crudo()
    validar_esquema(crudo)

    limpio, reporte = limpiar(crudo)

    if persistir:
        ruta = limpieza.guardar_procesado(limpio, ESPECIFICACION)
        reporte["archivo_procesado"] = ruta.name
        io.guardar_json(reporte, ESPECIFICACION.clave, "limpieza.json")

    return separar_variables(limpio)


def cargar_procesado() -> pd.DataFrame:
    """Carga el dataset limpio desde ``data/processed``."""
    return io.cargar_procesado(ESPECIFICACION.archivo_procesado)


__all__ = [
    "cargar_crudo",
    "cargar_procesado",
    "validar_esquema",
    "auditar",
    "describir",
    "limpiar",
    "separar_variables",
    "preparar",
]
