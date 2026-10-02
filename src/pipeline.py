"""Orquestador global del laboratorio.

Construye los **tres modelos** con un único comando: recorre cada modelo del
laboratorio ejecutando su flujo completo (comprensión de datos, limpieza,
entrenamiento, evaluación, exportación y gráficas) y, al terminar, comprueba que
los tres artefactos ``.joblib`` se hayan generado y se puedan cargar.

Este módulo no reimplementa nada: delega en el ``pipeline`` de cada modelo, que
es el que conoce sus propias fases. Así el comportamiento por modelo es
idéntico al de ejecutarlo por separado.

Uso:
    python -m src.pipeline                  # los tres modelos
    python -m src.pipeline --etapa modelar  # solo entrenamiento + exportación
    python -m src.pipeline --solo dolar     # un único modelo
"""

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass
from typing import Any

from .common import io

# Etapas admitidas, en el mismo orden que declaran los pipelines individuales.
ETAPAS: tuple[str, ...] = (
    "auditar",
    "limpiar",
    "modelar",
    "evaluar",
    "graficar",
    "completo",
)


@dataclass(frozen=True)
class Modelo:
    """Un modelo del laboratorio y el módulo que lo orquesta."""

    clave: str
    nombre: str
    orquestador: str
    especificacion: str


# Registro único de modelos. Añadir un modelo nuevo al laboratorio solo requiere
# una entrada aquí, más su paquete en ``src/<clave>/``.
MODELOS: tuple[Modelo, ...] = (
    Modelo(
        clave="dolar",
        nombre="Precio del Dólar",
        orquestador="src.dolar.pipeline",
        especificacion="src.dolar.especificacion",
    ),
    Modelo(
        clave="glucosa",
        nombre="Nivel de Glucosa",
        orquestador="src.glucosa.pipeline",
        especificacion="src.glucosa.especificacion",
    ),
    Modelo(
        clave="energia",
        nombre="Consumo de Energía",
        orquestador="src.energia.pipeline",
        especificacion="src.energia.especificacion",
    ),
)

CLAVES: tuple[str, ...] = tuple(modelo.clave for modelo in MODELOS)


def obtener(clave: str) -> Modelo:
    """Devuelve el modelo registrado con esa clave.

    Raises:
        KeyError: Si la clave no corresponde a ningún modelo del laboratorio.
    """
    for modelo in MODELOS:
        if modelo.clave == clave:
            return modelo
    raise KeyError(f"Modelo desconocido: '{clave}'. Válidos: {', '.join(CLAVES)}")


def _especificacion(clave: str) -> Any:
    """Carga la especificación declarada por el modelo."""
    modulo = importlib.import_module(obtener(clave).especificacion)
    return modulo.ESPECIFICACION


def construir(clave: str, etapa: str = "completo") -> None:
    """Ejecuta una etapa del modelo indicado.

    Se reutiliza el punto de entrada del propio ``pipeline`` del modelo, de modo
    que no exista lógica duplicada entre este orquestador y los individuales.
    """
    orquestador = importlib.import_module(obtener(clave).orquestador)

    # El pipeline lee su etapa de la línea de comandos. Se simula la invocación
    # y se restaura ``sys.argv`` para no alterar el proceso que nos llamó.
    argv_previo = sys.argv
    try:
        sys.argv = [f"src.{clave}.pipeline", "--etapa", etapa]
        orquestador.main()
    finally:
        sys.argv = argv_previo


def verificar(clave: str) -> dict[str, Any]:
    """Comprueba que el artefacto del modelo exista, cargue y prediga.

    Returns:
        Resumen con el estado del artefacto: existencia, número de variables y
        coeficientes, e intercepto.

    Raises:
        FileNotFoundError: Si el ``.joblib`` no se generó.
        ValueError: Si el artefacto no tiene la forma esperada.
    """
    especificacion = _especificacion(clave)
    estimador = io.cargar_modelo(especificacion)

    esperadas = len(especificacion.variables)
    if estimador.n_features_in_ != esperadas:
        raise ValueError(
            f"[{clave}] El modelo espera {estimador.n_features_in_} variables, "
            f"pero la especificación declara {esperadas}."
        )

    return {
        "clave": clave,
        "ruta": especificacion.ruta_modelo,
        "variables": estimador.n_features_in_,
        "coeficientes": [float(c) for c in estimador.coef_],
        "intercepto": float(estimador.intercept_),
    }


def metricas_guardadas(clave: str) -> dict[str, Any]:
    """Lee las métricas ya persistidas por la etapa de evaluación."""
    import json

    ruta = io.directorio_resultados(clave) / "metricas.json"
    if not ruta.is_file():
        return {}
    with ruta.open(encoding="utf-8") as manejador:
        return json.load(manejador)


def ejecutar(claves: tuple[str, ...] | None = None, etapa: str = "completo") -> int:
    """Construye los modelos indicados y verifica sus artefactos.

    Args:
        claves: Modelos a construir. Por defecto, todos.
        etapa: Etapa a ejecutar en cada modelo.

    Returns:
        Código de salida: ``0`` si todos los modelos se construyeron bien.
    """
    objetivo = claves if claves else CLAVES

    print("=" * 70)
    print(f"LABORATORIO DE MINERÍA DE DATOS — construyendo {len(objetivo)} modelo(s)")
    print(f"Etapa: {etapa}")
    print("=" * 70)

    fallos: list[tuple[str, str]] = []

    for clave in objetivo:
        try:
            construir(clave, etapa)
        except Exception as error:  # noqa: BLE001 - se reporta y se continúa
            fallos.append((clave, f"{type(error).__name__}: {error}"))
            print(f"\n[{clave}] ERROR: {type(error).__name__}: {error}", file=sys.stderr)
            continue

        # Las etapas parciales no tienen por qué producir el artefacto: solo se
        # verifica cuando el entrenamiento ha tenido lugar.
        if etapa in ("modelar", "completo"):
            try:
                estado = verificar(clave)
                print(f"[{clave}] Artefacto verificado: {estado['ruta'].name} "
                      f"({estado['variables']} variables, "
                      f"intercepto {estado['intercepto']:.4f})")
            except Exception as error:  # noqa: BLE001
                fallos.append((clave, f"{type(error).__name__}: {error}"))
                print(f"[{clave}] ERROR al verificar: {error}", file=sys.stderr)

    # --- Resumen final ---------------------------------------------------
    print()
    print("=" * 70)
    if etapa in ("modelar", "completo"):
        print("RESUMEN DE LOS MODELOS CONSTRUIDOS")
        print("=" * 70)
        for clave in objetivo:
            if any(clave == fallo for fallo, _ in fallos):
                continue
            metricas = metricas_guardadas(clave)
            prueba = metricas.get("metricas_prueba", {})
            if prueba:
                print(f"  {clave:<8} R² = {prueba.get('r2', float('nan')):.4f} | "
                      f"RMSE = {prueba.get('rmse', float('nan')):.4f} | "
                      f"modelo_{clave}.joblib")
            else:
                print(f"  {clave:<8} modelo_{clave}.joblib")
        print()

    if fallos:
        print(f"FINALIZADO CON {len(fallos)} ERROR(ES):")
        for clave, detalle in fallos:
            print(f"  - {clave}: {detalle}")
        print("=" * 70)
        return 1

    print(f"OK — {len(objetivo)} modelo(s) construido(s) correctamente.")
    print("=" * 70)
    return 0


def main() -> None:
    """Punto de entrada del orquestador global."""
    parser = argparse.ArgumentParser(
        description="Construye todos los modelos del laboratorio con un solo comando."
    )
    parser.add_argument(
        "--etapa",
        choices=ETAPAS,
        default="completo",
        help="Fase concreta a ejecutar en cada modelo (por defecto: completo).",
    )
    parser.add_argument(
        "--solo",
        choices=CLAVES,
        action="append",
        metavar="MODELO",
        help=(
            "Construye únicamente el modelo indicado. Se puede repetir "
            f"(--solo dolar --solo glucosa). Válidos: {', '.join(CLAVES)}."
        ),
    )
    args = parser.parse_args()

    claves = tuple(args.solo) if args.solo else None
    sys.exit(ejecutar(claves=claves, etapa=args.etapa))


if __name__ == "__main__":
    main()
