"""Especificación de las variables y de los modelos.

Define los tipos compartidos que describen **qué datos usa cada modelo**. Cada
modelo del laboratorio declara su propia especificación en su ``__init__.py``,
de modo que no hay definiciones duplicadas ni un fichero central que haya que
tocar al añadir un modelo nuevo.

Este módulo no contiene lógica de Machine Learning: es únicamente el contrato de
datos que consumen la limpieza, el modelado, la evaluación y la visualización.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from .paths import DATA_RAW_DIR, MODELS_DIR


@dataclass(frozen=True)
class EspecificacionVariable:
    """Metadatos de una variable: etiqueta, unidad y rango plausible.

    El rango se usa para **validar** los datos (detectar valores imposibles).
    Nunca se usa para recortar valores: eso falsearía el modelo.

    Attributes:
        nombre: Nombre de la columna en el dataset.
        etiqueta: Nombre legible para el informe.
        unidad: Unidad de medida.
        minimo: Valor mínimo plausible.
        maximo: Valor máximo plausible.
        entera: Indica si la variable es un recuento entero por naturaleza
            (día, edad, hora). Se declara explícitamente y **no** se deduce del
            rango: ``Inflacion`` va de 0 a 1 pero es una proporción continua.
    """

    nombre: str
    etiqueta: str
    unidad: str
    minimo: float
    maximo: float
    entera: bool = False

    def en_rango(self, valor: float) -> bool:
        """Indica si un valor cae dentro del rango plausible."""
        return self.minimo <= valor <= self.maximo


@dataclass(frozen=True)
class EspecificacionModelo:
    """Contrato de datos y artefactos de un modelo.

    Attributes:
        clave: Identificador del modelo (``"dolar"``, ``"glucosa"``,
            ``"energia"``). Coincide con el nombre de su paquete en ``src``.
        nombre: Nombre legible para el informe.
        descripcion: Objetivo del modelo en una frase.
        objetivo: Variable dependiente.
        variables: Variables independientes, en el orden de entrada.
        especificaciones: Metadatos de cada variable independiente.
        especificacion_objetivo: Metadatos de la variable objetivo.
        archivo_crudo: Nombre del CSV de origen en ``data/raw``.
        archivo_procesado: Nombre del CSV limpio en ``data/processed``.
        archivo_modelo: Nombre del artefacto exportado en ``models``.
    """

    clave: str
    nombre: str
    descripcion: str
    objetivo: str
    variables: tuple[str, ...]
    especificaciones: tuple[EspecificacionVariable, ...]
    especificacion_objetivo: EspecificacionVariable
    archivo_crudo: str
    archivo_procesado: str
    archivo_modelo: str

    @property
    def ruta_cruda(self):
        """Ruta absoluta al dataset crudo (nunca se modifica)."""
        return DATA_RAW_DIR / self.archivo_crudo

    @property
    def ruta_modelo(self):
        """Ruta absoluta al artefacto ``.joblib``."""
        return MODELS_DIR / self.clave / self.archivo_modelo

    @property
    def columnas(self) -> tuple[str, ...]:
        """Todas las columnas esperadas, con el objetivo al final."""
        return (*self.variables, self.objetivo)

    def especificacion_de(self, variable: str) -> EspecificacionVariable:
        """Devuelve los metadatos de una variable independiente.

        Raises:
            KeyError: Si la variable no pertenece al modelo.
        """
        for especificacion in self.especificaciones:
            if especificacion.nombre == variable:
                return especificacion
        raise KeyError(
            f"'{variable}' no es una variable del modelo '{self.clave}'. "
            f"Disponibles: {list(self.variables)}."
        )


# Número de variables independientes de todos los modelos del laboratorio.
NUMERO_VARIABLES: Final[int] = 3

__all__ = [
    "EspecificacionVariable",
    "EspecificacionModelo",
    "NUMERO_VARIABLES",
]
