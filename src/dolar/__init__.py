"""Modelo de predicción del precio del dólar.

Variables independientes: ``Dia``, ``Inflacion``, ``Tasa_interes``.
Variable objetivo: ``Precio_Dolar``.

Módulos:
    ``limpieza``        auditoría y limpieza reproducible
    ``modelo``          entrenamiento y exportación
    ``evaluacion``      métricas, coeficientes y análisis
    ``visualizacion``   gráficas del modelo
"""

from .especificacion import ESPECIFICACION

__all__ = ["ESPECIFICACION"]
