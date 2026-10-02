"""Modelo de predicción del nivel de glucosa.

Variables independientes: ``Edad``, ``IMC``, ``Actividad_Fisica``.
Variable objetivo: ``Nivel_Glucosa``.

Módulos:
    ``limpieza``        auditoría y limpieza reproducible
    ``modelo``          entrenamiento y exportación
    ``evaluacion``      métricas, coeficientes y análisis
    ``visualizacion``   gráficas del modelo
"""

from .especificacion import ESPECIFICACION

__all__ = ["ESPECIFICACION"]
