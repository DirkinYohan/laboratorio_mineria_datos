"""Orquestador del modelo de consumo de energía.

Encadena las fases del proyecto para este modelo:

    comprensión de datos -> limpieza -> preparación -> entrenamiento
    -> evaluación -> exportación -> visualización

Uso:
    python -m src.energia.pipeline                  # flujo completo
    python -m src.energia.pipeline --etapa auditar  # solo comprensión de datos
    python -m src.energia.pipeline --etapa limpiar  # solo limpieza
    python -m src.energia.pipeline --etapa modelar  # entrenamiento + evaluación
    python -m src.energia.pipeline --etapa evaluar  # solo evaluación
    python -m src.energia.pipeline --etapa graficar # solo gráficas
"""

from __future__ import annotations

from typing import Any

from src.common import io

from . import evaluacion, limpieza, modelo, visualizacion
from .especificacion import ESPECIFICACION


def comprender_datos() -> dict[str, Any]:
    """Fase 1: carga el dataset crudo y reporta su calidad."""
    df = limpieza.cargar_crudo()
    limpieza.validar_esquema(df)

    auditoria = limpieza.auditar(df)
    resumen: dict[str, Any] = {
        "modelo": ESPECIFICACION.clave,
        "nombre": ESPECIFICACION.nombre,
        "filas": int(df.shape[0]),
        "columnas": list(ESPECIFICACION.columnas),
        "nulos": io.contar_nulos(df, ESPECIFICACION.columnas),
        "duplicados": auditoria.duplicados,
        "perfil_columnas": [c.__dict__ for c in auditoria.perfil_columnas],
    }

    io.guardar_json(resumen, ESPECIFICACION.clave, "comprension_datos.json")
    io.guardar_json(auditoria.a_diccionario(), ESPECIFICACION.clave, "auditoria_cruda.json")
    io.guardar_tabla(
        limpieza.describir(df), ESPECIFICACION.clave, "estadisticas_crudas.csv"
    )
    return resumen


def preparar_datos() -> dict[str, Any]:
    """Fase 2: limpia el dataset y guarda la versión procesada."""
    X, y = limpieza.preparar(persistir=True)
    return {
        "filas_procesadas": int(X.shape[0]),
        "variables": list(X.columns),
        "objetivo": ESPECIFICACION.objetivo,
    }


def modelar() -> tuple[Any, Any]:
    """Fases 3-4: entrena, interpreta, evalúa y exporta el modelo."""
    entrenado = modelo.entrenar(persistir=True)
    modelo.guardar_interpretacion(entrenado)

    # La evaluación trabaja con el estimador exportado (mismo contrato que el
    # .joblib), de modo que valida exactamente lo que consumirá el frontend.
    evaluacion.guardar_reporte(entrenado.estimador)
    return entrenado, evaluacion.analizar(entrenado.estimador)


def graficar(entrenado: Any) -> list[str]:
    """Fase 5: genera todas las gráficas del modelo."""
    return [str(ruta) for ruta in visualizacion.generar_todas(entrenado)]


def ejecutar() -> None:
    """Ejecuta el flujo completo del modelo, fase por fase."""
    print(f"\n{'=' * 70}")
    print(f"[{ESPECIFICACION.clave}] {ESPECIFICACION.nombre}")
    print("=" * 70)

    print(f"[{ESPECIFICACION.clave}] 1. Comprensión de los datos ...")
    resumen = comprender_datos()
    print(f"[{ESPECIFICACION.clave}]    {resumen['filas']} filas, "
          f"{resumen['duplicados']} duplicados, "
          f"{sum(resumen['nulos'].values())} nulos")

    print(f"[{ESPECIFICACION.clave}] 2. Limpieza y preparación ...")
    preparacion = preparar_datos()
    print(f"[{ESPECIFICACION.clave}]    {preparacion['filas_procesadas']} filas procesadas")

    print(f"[{ESPECIFICACION.clave}] 3. Entrenamiento y evaluación ...")
    entrenado, analisis = modelar()
    metricas = analisis["evaluacion"]["metricas"]
    print(f"[{ESPECIFICACION.clave}]    {entrenado.ecuacion()}")
    print(f"[{ESPECIFICACION.clave}]    R2 = {metricas['r2']:.6f} | "
          f"MSE = {metricas['mse']:.4f} | RMSE = {metricas['rmse']:.4f}")

    print(f"[{ESPECIFICACION.clave}] 4. Análisis ...")
    print(f"[{ESPECIFICACION.clave}]    {analisis['interpretacion_desempeno']}")
    print(f"[{ESPECIFICACION.clave}]    Mayor impacto: {entrenado.mas_influyente.nombre}")

    print(f"[{ESPECIFICACION.clave}] 5. Exportación ...")
    print(f"[{ESPECIFICACION.clave}]    {ESPECIFICACION.ruta_modelo}")

    print(f"[{ESPECIFICACION.clave}] 6. Visualizaciones ...")
    for ruta in graficar(entrenado):
        print(f"[{ESPECIFICACION.clave}]    {ruta}")

    print(f"[{ESPECIFICACION.clave}] Flujo completado.")


def main() -> None:
    """Punto de entrada del orquestador."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Flujo completo del modelo de consumo de energía."
    )
    parser.add_argument(
        "--etapa",
        choices=("auditar", "limpiar", "modelar", "evaluar", "graficar", "completo"),
        default="completo",
        help="Fase concreta a ejecutar (por defecto: completo).",
    )
    args = parser.parse_args()

    if args.etapa == "auditar":
        print(comprender_datos())
    elif args.etapa == "limpiar":
        print(preparar_datos())
    elif args.etapa == "modelar":
        _, analisis = modelar()
        print(analisis["interpretacion_desempeno"])
    elif args.etapa == "evaluar":
        print(evaluacion.analizar()["interpretacion_desempeno"])
    elif args.etapa == "graficar":
        entrenado = modelo.entrenar(persistir=False)
        for ruta in graficar(entrenado):
            print(ruta)
    else:
        ejecutar()


if __name__ == "__main__":
    main()
