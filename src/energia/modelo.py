"""Entrenamiento del modelo de consumo de energía.

El algoritmo es el mismo en los tres modelos: se delega en la lógica genérica de
``src.common.modelado``. Este módulo solo resuelve qué datos y qué especificación
usar, y añade la interpretación propia del dominio.

Uso:
    python -m src.energia.modelo
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common import io
from src.common.metricas import calcular_metricas, interpretar_coeficiente
from src.common.modelado import ModeloEntrenado, entrenar_regresion

from . import limpieza
from .especificacion import ESPECIFICACION

NOTA_METODOLOGICA = (
    "Las tres variables resultaron estadísticamente significativas (p < 0,001). "
    "'Temperatura' presenta el mayor impacto sobre el consumo según el "
    "coeficiente estandarizado, seguida de 'Hora' y 'Dia_Semana'. 'Hora' y "
    "'Dia_Semana' se mantienen en escala lineal: se verificó que la codificación "
    "cíclica seno/coseno reduce el R² de prueba, porque 'Hora' es una rampa "
    "monótona y no un ciclo horario. Los VIF cercanos a 1 descartan "
    "multicolinealidad."
)


def entrenar(*, persistir: bool = True) -> ModeloEntrenado:
    """Entrena el modelo sobre el dataset ya preparado.

    Args:
        persistir: Si es ``True``, exporta el artefacto ``.joblib``, la tabla de
            coeficientes y el resumen del modelo.

    Returns:
        El modelo entrenado con su inferencia estadística y sus métricas.
    """
    X, y = limpieza.preparar(persistir=persistir)

    modelo = entrenar_regresion(
        X,
        y,
        variables=ESPECIFICACION.variables,
        objetivo=ESPECIFICACION.objetivo,
        funcion_metricas=calcular_metricas,
        nota_metodologica=NOTA_METODOLOGICA,
    )

    if persistir:
        io.guardar_modelo(modelo.estimador, ESPECIFICACION)
        io.guardar_tabla(
            modelo.tabla_coeficientes(), ESPECIFICACION.clave, "coeficientes.csv"
        )
        io.guardar_json(modelo.resumen(), ESPECIFICACION.clave, "modelo.json")

    return modelo


def interpretar(modelo: ModeloEntrenado) -> list[dict[str, Any]]:
    """Interpreta cada coeficiente en las unidades del problema.

    Returns:
        Una lista ordenada por impacto (β estandarizado en valor absoluto).
    """
    especificacion_objetivo = ESPECIFICACION.especificacion_objetivo
    interpretaciones: list[dict[str, Any]] = []

    for coeficiente in modelo.coeficientes:
        variable = ESPECIFICACION.especificacion_de(coeficiente.nombre)
        interpretaciones.append(
            {
                "variable": coeficiente.nombre,
                "etiqueta": variable.etiqueta,
                "coeficiente": coeficiente.estimacion,
                "p_valor": coeficiente.valor_p,
                "beta_estandarizado": coeficiente.beta_estandarizado,
                "significativo": coeficiente.significativo,
                "significancia": coeficiente.etiqueta_significancia,
                "interpretacion": interpretar_coeficiente(
                    variable.etiqueta,
                    coeficiente.estimacion,
                    variable.unidad,
                    especificacion_objetivo.etiqueta,
                    especificacion_objetivo.unidad,
                ),
            }
        )

    return sorted(
        interpretaciones, key=lambda item: abs(item["beta_estandarizado"]), reverse=True
    )


def guardar_interpretacion(modelo: ModeloEntrenado) -> Path:
    """Persiste la interpretación y el ranking de impacto de las variables."""
    payload = {
        "modelo": ESPECIFICACION.clave,
        "ecuacion": modelo.ecuacion(),
        "intercepto": modelo.intercepto,
        "criterio_mayor_impacto": (
            "Se compara el valor absoluto del coeficiente estandarizado (β). "
            "El coeficiente bruto no es comparable entre variables porque sus "
            "escalas difieren (°C, hora y día de la semana)."
        ),
        "variable_mayor_impacto": modelo.mas_influyente.nombre,
        "ranking_impacto": [
            {
                "posicion": posicion,
                "variable": item["variable"],
                "beta_estandarizado": item["beta_estandarizado"],
            }
            for posicion, item in enumerate(interpretar(modelo), start=1)
        ],
        "interpretaciones": interpretar(modelo),
    }
    return io.guardar_json(payload, ESPECIFICACION.clave, "interpretacion.json")


def main() -> None:
    """Punto de entrada: ``python -m src.energia.modelo``."""
    modelo = entrenar()
    guardar_interpretacion(modelo)

    print(f"[{ESPECIFICACION.clave}] Modelo entrenado en {ESPECIFICACION.ruta_modelo}")
    print(f"[{ESPECIFICACION.clave}] {modelo.ecuacion()}")
    print(modelo.tabla_coeficientes().to_string(index=False))
    print(f"[{ESPECIFICACION.clave}] R2 prueba  = {modelo.metricas_prueba['r2']:.6f}")
    print(f"[{ESPECIFICACION.clave}] RMSE prueba = {modelo.metricas_prueba['rmse']:.4f}")
    print(
        f"[{ESPECIFICACION.clave}] Mayor impacto = {modelo.mas_influyente.nombre}"
    )


if __name__ == "__main__":
    main()
