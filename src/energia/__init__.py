"""Modelo de predicción del consumo de energía.

Variables independientes: ``Temperatura``, ``Hora``, ``Dia_Semana``.
Variable objetivo: ``Consumo_Energia``.

Módulos:
    ``limpieza``        auditoría, limpieza y codificación temporal
    ``modelo``          entrenamiento y exportación
    ``evaluacion``      métricas, coeficientes y análisis
    ``visualizacion``   gráficas del modelo
"""

from .especificacion import ESPECIFICACION

__all__ = ["ESPECIFICACION"]
