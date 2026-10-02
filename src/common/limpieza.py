"""Auditoría de calidad y limpieza reproducible de los datasets.

Todas las funciones son genéricas: reciben una ``EspecificacionModelo`` y
trabajan igual para los tres modelos. La lógica de limpieza se escribe **una
sola vez** y se reutiliza.

Principios aplicados
--------------------
1. ``data/raw`` es de solo lectura: nunca se sobrescribe.
2. Los valores atípicos se **detectan y documentan**, pero solo se eliminan si
   existe una justificación verificable (valor imposible, duplicado o nulo).
   Eliminar un valor extremo pero plausible sesgaría el modelo.
3. Toda decisión de limpieza deja un reporte en ``results/<modelo>/``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .especificacion import EspecificacionModelo
from .paths import DATA_PROCESSED_DIR


# --- Reporte de auditoría ----------------------------------------------
@dataclass
class AuditoriaColumna:
    """Perfil de calidad de una columna."""

    nombre: str
    tipo: str
    nulos: int
    unicos: int
    atipicos_iqr: int
    minimo: float
    maximo: float
    media: float
    desviacion: float
    asimetria: float
    fuera_de_rango: int


@dataclass
class Auditoria:
    """Reporte de auditoría de un dataset completo."""

    clave_modelo: str
    filas: int
    columnas: int
    duplicados: int
    nulos_totales: int
    perfil_columnas: list[AuditoriaColumna] = field(default_factory=list)

    def a_diccionario(self) -> dict[str, Any]:
        """Convierte el reporte a un diccionario serializable a JSON."""
        return {
            "modelo": self.clave_modelo,
            "filas": self.filas,
            "columnas": self.columnas,
            "duplicados": self.duplicados,
            "nulos_totales": self.nulos_totales,
            "perfil_columnas": [asdict(columna) for columna in self.perfil_columnas],
        }


# --- Detección ----------------------------------------------------------
def detectar_duplicados(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve las filas duplicadas exactas (sin la primera aparición)."""
    return df[df.duplicated(keep="first")]


def detectar_nulos(df: pd.DataFrame) -> pd.Series:
    """Cuenta los valores nulos por columna."""
    return df.isna().sum()


def detectar_atipicos_iqr(serie: pd.Series, factor: float = 1.5) -> pd.Series:
    """Marca los valores atípicos de una serie por el criterio del rango IQR.

    Un valor es atípico si cae fuera de ``[Q1 - factor*IQR, Q3 + factor*IQR]``.

    Args:
        serie: Serie numérica a analizar.
        factor: Multiplicador del IQR (1.5 = moderado, 3.0 = extremo).

    Returns:
        Máscara booleana alineada con ``serie``.
    """
    q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
    rango = q3 - q1
    return (serie < q1 - factor * rango) | (serie > q3 + factor * rango)


def detectar_atipicos_z(serie: pd.Series, umbral: float = 3.0) -> pd.Series:
    """Marca los valores atípicos de una serie por puntuación Z."""
    desviacion = serie.std(ddof=0)
    if desviacion == 0 or np.isnan(desviacion):
        return pd.Series(False, index=serie.index)
    return ((serie - serie.mean()) / desviacion).abs() > umbral


def reporte_atipicos(
    df: pd.DataFrame, columnas: Sequence[str], factor: float = 1.5
) -> pd.DataFrame:
    """Resume los atípicos detectados en varias columnas.

    Returns:
        ``DataFrame`` con una fila por columna y el número de atípicos por
        criterio IQR y Z, junto con los límites calculados.
    """
    filas: list[dict[str, Any]] = []
    for columna in columnas:
        serie = df[columna]
        q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
        rango = q3 - q1
        filas.append(
            {
                "variable": columna,
                "atipicos_iqr": int(detectar_atipicos_iqr(serie, factor).sum()),
                "atipicos_z3": int(detectar_atipicos_z(serie, 3.0).sum()),
                "limite_inferior": q1 - factor * rango,
                "limite_superior": q3 + factor * rango,
                "minimo": serie.min(),
                "maximo": serie.max(),
            }
        )
    return pd.DataFrame(filas)


def comprobar_rangos(
    df: pd.DataFrame, especificacion: EspecificacionModelo
) -> dict[str, list[int]]:
    """Comprueba que los valores caigan en rangos plausibles de dominio.

    Returns:
        ``{variable: [indices fuera de rango]}``; solo incluye las variables con
        al menos un valor fuera de rango.
    """
    infractoras: dict[str, list[int]] = {}
    for variable in especificacion.especificaciones:
        mascara = ~df[variable.nombre].between(variable.minimo, variable.maximo)
        if mascara.any():
            infractoras[variable.nombre] = df.index[mascara].tolist()

    objetivo = especificacion.especificacion_objetivo
    mascara = ~df[especificacion.objetivo].between(objetivo.minimo, objetivo.maximo)
    if mascara.any():
        infractoras[especificacion.objetivo] = df.index[mascara].tolist()

    return infractoras


def auditar(df: pd.DataFrame, especificacion: EspecificacionModelo) -> Auditoria:
    """Genera el reporte completo de calidad de un dataset crudo."""
    infractoras = comprobar_rangos(df, especificacion)
    columnas: list[AuditoriaColumna] = []

    for columna in especificacion.columnas:
        serie = df[columna]
        columnas.append(
            AuditoriaColumna(
                nombre=columna,
                tipo=str(serie.dtype),
                nulos=int(serie.isna().sum()),
                unicos=int(serie.nunique()),
                atipicos_iqr=int(detectar_atipicos_iqr(serie).sum()),
                minimo=float(serie.min()),
                maximo=float(serie.max()),
                media=float(serie.mean()),
                desviacion=float(serie.std()),
                asimetria=float(serie.skew()),
                fuera_de_rango=len(infractoras.get(columna, [])),
            )
        )

    return Auditoria(
        clave_modelo=especificacion.clave,
        filas=int(df.shape[0]),
        columnas=int(df.shape[1]),
        duplicados=int(df.duplicated().sum()),
        nulos_totales=int(df.isna().sum().sum()),
        perfil_columnas=columnas,
    )


# --- Transformación -----------------------------------------------------
def seleccionar_columnas(
    df: pd.DataFrame, especificacion: EspecificacionModelo
) -> pd.DataFrame:
    """Conserva únicamente las variables usadas por el modelo.

    Deja las variables independientes primero y la objetivo al final, en el
    orden declarado por la especificación.
    """
    return df.loc[:, list(especificacion.columnas)].copy()


def ajustar_tipos(
    df: pd.DataFrame, especificacion: EspecificacionModelo
) -> pd.DataFrame:
    """Fuerza tipos numéricos coherentes con el modelo.

    Solo se convierten a entero las variables marcadas explícitamente como
    ``entera`` en la especificación (recuentos como día, edad u hora). El resto
    queda en punto flotante.

    Nota: no se deduce del rango. ``Inflacion`` va de 0 a 1 pero es una
    proporción continua, no un recuento.
    """
    resultado = df.copy()
    variables_enteras = {
        variable.nombre
        for variable in especificacion.especificaciones
        if variable.entera
    }
    for columna in resultado.columns:
        if columna in variables_enteras:
            resultado[columna] = resultado[columna].round().astype("int64")
        else:
            resultado[columna] = resultado[columna].astype("float64")
    return resultado


def eliminar_filas_invalidas(
    df: pd.DataFrame, especificacion: EspecificacionModelo
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Elimina filas no utilizables y documenta cuántas eliminó.

    Criterios:
        - filas con valores nulos en las variables del modelo,
        - filas duplicadas exactas,
        - filas con valores fuera del rango plausible (imposibles).

    Returns:
        ``(dataset_limpio, {"nulos": n, "duplicados": n, "fuera_de_rango": n})``
    """
    reporte = {"nulos": 0, "duplicados": 0, "fuera_de_rango": 0}
    resultado = df.copy()

    antes = len(resultado)
    resultado = resultado.dropna(subset=list(especificacion.columnas))
    reporte["nulos"] = antes - len(resultado)

    antes = len(resultado)
    resultado = resultado.drop_duplicates(keep="first")
    reporte["duplicados"] = antes - len(resultado)

    infractoras = comprobar_rangos(resultado, especificacion)
    if infractoras:
        indices = sorted({i for filas in infractoras.values() for i in filas})
        antes = len(resultado)
        resultado = resultado.drop(index=indices)
        reporte["fuera_de_rango"] = antes - len(resultado)

    return resultado.reset_index(drop=True), reporte


def guardar_procesado(df: pd.DataFrame, especificacion: EspecificacionModelo):
    """Guarda el dataset limpio en ``data/processed/<modelo>_clean.csv``.

    Returns:
        La ruta del archivo escrito.
    """
    destino = DATA_PROCESSED_DIR / especificacion.archivo_procesado
    destino.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destino, index=False)
    return destino


def resumen_limpieza(
    especificacion: EspecificacionModelo,
    filas_antes: int,
    eliminaciones: dict[str, int],
    filas_despues: int,
) -> dict[str, Any]:
    """Construye el reporte de limpieza aplicado a un dataset."""
    return {
        "modelo": especificacion.clave,
        "filas_antes": int(filas_antes),
        "filas_despues": int(filas_despues),
        "filas_eliminadas": int(filas_antes - filas_despues),
        "detalle_eliminacion": eliminaciones,
        "criterio_atipicos": (
            "Los valores atípicos se detectan con el criterio IQR (1,5) y se "
            "documentan en la auditoría. No se eliminan porque ninguno resultó "
            "imposible: todos caen dentro de rangos plausibles del dominio, por "
            "lo que representan variabilidad real del fenómeno."
        ),
    }


def describir(df: pd.DataFrame, columnas: Iterable[str]) -> pd.DataFrame:
    """Estadística descriptiva de las columnas indicadas."""
    return df[list(columnas)].describe().T


__all__ = [
    "AuditoriaColumna",
    "Auditoria",
    "detectar_duplicados",
    "detectar_nulos",
    "detectar_atipicos_iqr",
    "detectar_atipicos_z",
    "reporte_atipicos",
    "comprobar_rangos",
    "auditar",
    "seleccionar_columnas",
    "ajustar_tipos",
    "eliminar_filas_invalidas",
    "guardar_procesado",
    "resumen_limpieza",
    "describir",
]
