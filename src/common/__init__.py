"""Lógica compartida por los tres modelos de Machine Learning.

Contiene la infraestructura común que no depende de un dataset concreto:
rutas, contrato de datos, auditoría/limpieza, entrada-salida, modelado,
métricas y visualización genéricas.

Ningún módulo de esta capa contiene lógica de interfaz de usuario.
"""

from .especificacion import (
    NUMERO_VARIABLES,
    EspecificacionModelo,
    EspecificacionVariable,
)

__all__ = [
    "EspecificacionVariable",
    "EspecificacionModelo",
    "NUMERO_VARIABLES",
]
