"""Especificación de datos del modelo de consumo de energía.

Este es el único lugar donde se declaran las variables del modelo: el resto de
módulos lo importan desde aquí, de modo que no hay nombres de columna duplicados
por el código.
"""

from __future__ import annotations

from src.common.especificacion import EspecificacionModelo, EspecificacionVariable

CLAVE: str = "energia"

ESPECIFICACION: EspecificacionModelo = EspecificacionModelo(
    clave=CLAVE,
    nombre="Consumo de Energía",
    descripcion=(
        "Predice el consumo de energía a partir de la temperatura, la hora y el "
        "día de la semana."
    ),
    objetivo="Consumo_Energia",
    variables=("Temperatura", "Hora", "Dia_Semana"),
    especificaciones=(
        EspecificacionVariable(
            nombre="Temperatura",
            etiqueta="Temperatura",
            unidad="°C",
            minimo=-30,
            maximo=60,
        ),
        EspecificacionVariable(
            nombre="Hora",
            etiqueta="Hora",
            unidad="hora (1-24)",
            minimo=1,
            maximo=24,
            entera=True,
        ),
        EspecificacionVariable(
            nombre="Dia_Semana",
            etiqueta="Día de la semana",
            unidad="día (1-7)",
            minimo=1,
            maximo=7,
            entera=True,
        ),
    ),
    especificacion_objetivo=EspecificacionVariable(
        nombre="Consumo_Energia",
        etiqueta="Consumo de energía",
        unidad="kWh",
        minimo=0,
        maximo=2000,
    ),
    archivo_crudo="energia_data.csv",
    archivo_procesado="energia_clean.csv",
    archivo_modelo="modelo_energia.joblib",
)

__all__ = ["CLAVE", "ESPECIFICACION"]
