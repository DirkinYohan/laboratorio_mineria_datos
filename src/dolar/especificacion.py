"""Especificación de datos del modelo de precio del dólar.

Especificación de datos del modelo de precio del dólar. Este es el único lugar
donde se declaran las variables del modelo: el resto de módulos lo importan
desde aquí, de modo que no hay nombres de columna duplicados por el código.
"""

from __future__ import annotations

from src.common.especificacion import EspecificacionModelo, EspecificacionVariable

CLAVE: str = "dolar"

ESPECIFICACION: EspecificacionModelo = EspecificacionModelo(
    clave=CLAVE,
    nombre="Precio del Dólar",
    descripcion=(
        "Predice el precio del dólar a partir del día, la inflación y la tasa "
        "de interés."
    ),
    objetivo="Precio_Dolar",
    variables=("Dia", "Inflacion", "Tasa_interes"),
    especificaciones=(
        EspecificacionVariable(
            nombre="Dia",
            etiqueta="Día",
            unidad="días",
            minimo=1,
            maximo=3650,
            entera=True,
        ),
        EspecificacionVariable(
            nombre="Inflacion",
            etiqueta="Inflación",
            unidad="fracción decimal",
            minimo=0,
            maximo=1,
        ),
        EspecificacionVariable(
            nombre="Tasa_interes",
            etiqueta="Tasa de interés",
            unidad="porcentaje",
            minimo=0,
            maximo=100,
        ),
    ),
    especificacion_objetivo=EspecificacionVariable(
        nombre="Precio_Dolar",
        etiqueta="Precio del dólar",
        unidad="unidades monetarias",
        minimo=1,
        maximo=100000,
    ),
    archivo_crudo="dolar_data.csv",
    archivo_procesado="dolar_clean.csv",
    archivo_modelo="modelo_dolar.joblib",
)

__all__ = ["CLAVE", "ESPECIFICACION"]
