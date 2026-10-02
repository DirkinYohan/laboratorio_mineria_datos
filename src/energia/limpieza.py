"""Limpieza y preparación de los datos del modelo de consumo de energía.

Pipeline reproducible, sin intervención manual:

    cargar crudo -> auditar -> seleccionar columnas -> tipar
    -> eliminar filas inválidas -> codificar variables temporales
    -> separar X e y -> guardar en data/processed

Decisión sobre las variables temporales
----------------------------------------
``Hora`` y ``Dia_Semana`` son variables temporales. Aunque la convención sería
codificarlas de forma «ciclada» (seno y coseno), la evidencia obtenida de los
datos indica que:

1. ``Hora`` **no es cíclica en este dataset**: sus valores recorren 1→24 de
   forma monótona y se repiten en bucle (los incrementos son siempre +1, salvo
   el reinicio de 24 a 1). El consumo **crece de forma monótona** con la hora
   (de ≈348 a ≈459 de media), sin el patrón de dos picos propio de un ciclo
   diario.
2. Codificar ``Hora`` con seno/coseno **empeora** el ajuste de forma drástica:
   el R² de prueba cae de ≈0,90 a ≈0,79, porque una onda periódica no puede
   representar una rampa monótona.

Por tanto se conserva la codificación **lineal**, decisión que queda documentada
y que se puede reproducir con ``comparar_codificaciones_hora``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.common import io, limpieza
from src.common.limpieza import Auditoria
from .especificacion import ESPECIFICACION


def cargar_crudo() -> pd.DataFrame:
    """Carga el dataset crudo ``data/raw/energia_data.csv``."""
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


def codificacion_ciclica(serie: pd.Series, periodo: int) -> tuple[pd.Series, pd.Series]:
    """Codifica una variable periódica con seno y coseno.

    Args:
        serie: Valores de la variable (por ejemplo, la hora 1-24).
        periodo: Longitud del ciclo (24 para horas, 7 para días).

    Returns:
        ``(seno, coseno)`` como series alineadas con ``serie``.
    """
    angulo = 2.0 * np.pi * serie / periodo
    return np.sin(angulo), np.cos(angulo)


def comparar_codificaciones_hora(df: pd.DataFrame) -> dict[str, float]:
    """Compara empíricamente la codificación lineal y la cíclica de ``Hora``.

    Ajusta un modelo con cada alternativa usando la misma partición de datos y
    devuelve el R² de prueba de cada una. Sustenta con evidencia la decisión de
    codificación.

    Returns:
        ``{"hora_lineal", "hora_ciclica", "diferencia"}``.
    """
    from sklearn.linear_model import LinearRegression
    from src.common.modelado import dividir_datos

    objetivo = df[ESPECIFICACION.objetivo].to_numpy(dtype=float)
    temperatura = df["Temperatura"].to_numpy(dtype=float)
    hora = df["Hora"].to_numpy(dtype=float)
    dia = df["Dia_Semana"].to_numpy(dtype=float)

    seno = np.sin(2.0 * np.pi * hora / 24.0)
    coseno = np.cos(2.0 * np.pi * hora / 24.0)

    variantes = {
        "hora_lineal": np.column_stack([temperatura, hora, dia]),
        "hora_ciclica": np.column_stack([temperatura, seno, coseno, dia]),
    }

    puntuaciones: dict[str, float] = {}
    for etiqueta, matriz in variantes.items():
        X_ent, X_prueba, y_ent, y_prueba = dividir_datos(
            pd.DataFrame(matriz), pd.Series(objetivo)
        )
        modelo = LinearRegression().fit(X_ent, y_ent)
        puntuaciones[etiqueta] = float(modelo.score(X_prueba, y_prueba))

    puntuaciones["diferencia"] = (
        puntuaciones["hora_lineal"] - puntuaciones["hora_ciclica"]
    )
    return puntuaciones


def codificar_temporales(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara las variables temporales para el modelo.

    Como la codificación elegida es la **lineal** (véase el docstring del
    módulo), no se requiere ninguna transformación: ``Hora`` se conserva en el
    rango 1-24 y ``Dia_Semana`` en 1-7. La conversión a punto flotante la
    realiza internamente el estimador, por lo que aquí se preservan los tipos
    enteros del dataset limpio.

    La función se mantiene separada para poder cambiar la codificación de forma
    explícita en el futuro.
    """
    return df.copy()


def limpiar(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Limpia el dataset crudo y documenta lo que hizo.

    Operaciones aplicadas (comprobadas sobre los datos reales):

    1. Se conservan únicamente las cuatro variables del modelo.
    2. Se fijan los tipos: ``Hora`` y ``Dia_Semana`` enteros; el resto en punto
       flotante.
    3. Se eliminan filas con nulos, duplicados exactos y valores imposibles.
    4. Los valores atípicos de temperatura y consumo se detectan y se
       documentan, pero **no se eliminan**: son temperaturas extremas plausibles
       (hasta ≈44,6 °C) con su consumo correspondiente, y el modelo debe aprender
       a predecirlas en lugar de ignorarlas.

    Args:
        df: Dataset crudo ya cargado.

    Returns:
        ``(dataset_limpio, reporte)`` con el detalle de la limpieza.
    """
    filas_antes = len(df)

    trabajo = limpieza.seleccionar_columnas(df, ESPECIFICACION)
    trabajo = limpieza.ajustar_tipos(trabajo, ESPECIFICACION)
    trabajo, eliminaciones = limpieza.eliminar_filas_invalidas(trabajo, ESPECIFICACION)

    comparacion = comparar_codificaciones_hora(trabajo)
    trabajo = codificar_temporales(trabajo)

    reporte = limpieza.resumen_limpieza(
        ESPECIFICACION, filas_antes, eliminaciones, len(trabajo)
    )
    reporte["codificacion_temporal"] = {
        "decision": "Hora y Dia_Semana en escala lineal",
        "r2_hora_lineal": comparacion["hora_lineal"],
        "r2_hora_ciclica": comparacion["hora_ciclica"],
        "diferencia_r2": comparacion["diferencia"],
        "justificacion": (
            "'Hora' no es cíclica en estos datos: sus valores recorren 1-24 de "
            "forma monótona y el consumo crece monótonamente con ella. "
            "Codificarla con seno/coseno reduce el R² de prueba frente a la "
            "codificación lineal, por lo que se conserva la escala original."
        ),
    }
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
            ``data/processed`` y el reporte en ``results/energia``.

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
    "codificacion_ciclica",
    "comparar_codificaciones_hora",
    "codificar_temporales",
    "limpiar",
    "separar_variables",
    "preparar",
]
