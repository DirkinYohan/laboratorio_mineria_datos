"""Especificación de datos del modelo de nivel de glucosa.

Este es el único lugar donde se declaran las variables del modelo: el resto de
módulos lo importan desde aquí, de modo que no hay nombres de columna duplicados
por el código.
"""

from __future__ import annotations

from src.common.especificacion import EspecificacionModelo, EspecificacionVariable

CLAVE: str = "glucosa"

ESPECIFICACION: EspecificacionModelo = EspecificacionModelo(
    clave=CLAVE,
    nombre="Nivel de Glucosa",
    descripcion=(
        "Predice el nivel de glucosa en sangre a partir de la edad, el IMC y la "
        "actividad física."
    ),
    objetivo="Nivel_Glucosa",
    variables=("Edad", "IMC", "Actividad_Fisica"),
    especificaciones=(
        EspecificacionVariable(
            nombre="Edad",
            etiqueta="Edad",
            unidad="años",
            minimo=0,
            maximo=120,
            entera=True,
        ),
        EspecificacionVariable(
            nombre="IMC",
            etiqueta="IMC",
            unidad="kg/m²",
            minimo=5,
            maximo=80,
        ),
        EspecificacionVariable(
            nombre="Actividad_Fisica",
            etiqueta="Actividad física",
            unidad="nivel (0-9)",
            minimo=0,
            maximo=9,
            entera=True,
        ),
    ),
    especificacion_objetivo=EspecificacionVariable(
        nombre="Nivel_Glucosa",
        etiqueta="Nivel de glucosa",
        unidad="mg/dL",
        minimo=20,
        maximo=600,
    ),
    archivo_crudo="glucosa_data.csv",
    archivo_procesado="glucosa_clean.csv",
    archivo_modelo="modelo_glucosa.joblib",
)

__all__ = ["CLAVE", "ESPECIFICACION"]
