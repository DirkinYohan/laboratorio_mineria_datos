"""Utilidades de visualización compartidas.

Centraliza el estilo, la creación de figuras y la exportación de imágenes, de
modo que las gráficas de los tres modelos sean homogéneas. Incluye las gráficas
genéricas que se repiten:

- dispersión de cada variable independiente frente al objetivo,
- mapa de calor de correlación,
- valores reales frente a predichos,
- distribución de residuos,
- coeficientes estandarizados.

Las gráficas específicas del dominio de cada modelo se construyen en
``src/<modelo>/visualizacion.py`` reutilizando estos helpers.

Todas las gráficas son análisis de Machine Learning: no hay ningún componente de
interfaz de usuario.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from . import io

TAMANO_PREDETERMINADO: tuple[float, float] = (8.0, 5.0)
DPI_PREDETERMINADO: int = 150


def obtener_pyplot() -> Any:
    """Devuelve ``matplotlib.pyplot`` con backend no interactivo ``Agg``.

    Se importa de forma perezosa y con backend ``Agg`` para que los scripts
    puedan ejecutarse en entornos sin pantalla (CI, servidores) y guardar las
    figuras directamente en disco.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def configurar_estilo() -> None:
    """Aplica un estilo visual consistente a todas las gráficas del proyecto."""
    plt = obtener_pyplot()
    plt.rcParams.update(
        {
            "figure.figsize": TAMANO_PREDETERMINADO,
            "figure.dpi": DPI_PREDETERMINADO,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )


def nueva_figura(
    *,
    titulo: str | None = None,
    etiqueta_x: str | None = None,
    etiqueta_y: str | None = None,
    tamano: tuple[float, float] = TAMANO_PREDETERMINADO,
) -> tuple[Any, Any]:
    """Crea una figura con ejes y etiquetas opcionales.

    Returns:
        La tupla ``(figura, ejes)``.
    """
    configurar_estilo()
    plt = obtener_pyplot()
    figura, ejes = plt.subplots(figsize=tamano)

    if titulo:
        ejes.set_title(titulo)
    if etiqueta_x:
        ejes.set_xlabel(etiqueta_x)
    if etiqueta_y:
        ejes.set_ylabel(etiqueta_y)

    return figura, ejes


def exportar(figura: Any, clave_modelo: str, nombre: str, dpi: int = DPI_PREDETERMINADO) -> Path:
    """Exporta una figura a ``visualizations/<modelo>/``.

    Returns:
        La ruta del archivo escrito.
    """
    return io.guardar_figura(figura, clave_modelo, nombre, dpi=dpi)


def cerrar(figura: Any) -> None:
    """Libera la memoria asociada a una figura."""
    obtener_pyplot().close(figura)


# --- Gráficas reutilizables --------------------------------------------
def dispersion_frente_objetivo(
    df: Any,
    variable: str,
    objetivo: str,
    clave_modelo: str,
    *,
    etiqueta_variable: str,
    etiqueta_objetivo: str,
    nombre_archivo: str,
) -> Path:
    """Dispersión de una variable independiente frente a la variable objetivo.

    Añade la recta de tendencia estimada por mínimos cuadrados para visualizar
    el sentido y la intensidad de la relación.
    """
    import numpy as np

    plt = obtener_pyplot()
    figura, ejes = nueva_figura(
        titulo=f"{etiqueta_variable} vs. {etiqueta_objetivo}",
        etiqueta_x=f"{etiqueta_variable} ({variable})",
        etiqueta_y=f"{etiqueta_objetivo} ({objetivo})",
    )
    ejes.scatter(
        df[variable], df[objetivo], s=12, alpha=0.45, color="#1f77b4",
        edgecolors="none", label="Observaciones",
    )

    pendiente, intercepto = np.polyfit(df[variable], df[objetivo], 1)
    xs = np.linspace(df[variable].min(), df[variable].max(), 100)
    ejes.plot(
        xs, pendiente * xs + intercepto, color="#d62728", linewidth=2,
        label=f"Tendencia (pendiente = {pendiente:.4f})",
    )
    ejes.legend()

    ruta = exportar(figura, clave_modelo, nombre_archivo)
    cerrar(figura)
    return ruta


def mapa_calor_correlacion(
    df: Any,
    columnas: Sequence[str],
    clave_modelo: str,
    *,
    nombre_archivo: str = "matriz_correlacion.png",
    titulo: str = "Matriz de correlación",
) -> Path:
    """Mapa de calor con la matriz de correlación de las variables del modelo."""
    plt = obtener_pyplot()
    seaborn = __import__("seaborn")

    configurar_estilo()
    correlacion = df[list(columnas)].corr(numeric_only=True)

    figura, ejes = plt.subplots(figsize=(6.5, 5.5))
    seaborn.heatmap(
        correlacion,
        annot=True,
        fmt=".4f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ejes,
    )
    ejes.set_title(titulo)
    plt.setp(ejes.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ejes.get_yticklabels(), rotation=0)

    ruta = exportar(figura, clave_modelo, nombre_archivo)
    cerrar(figura)
    return ruta


def real_frente_predicho(
    y_real: Any,
    y_predicho: Any,
    clave_modelo: str,
    *,
    etiqueta_objetivo: str,
    nombre_archivo: str = "real_vs_predicho.png",
    muestra_maxima: int = 1500,
) -> Path:
    """Dispersión de valores reales frente a predichos.

    La recta diagonal representa la predicción perfecta. Si se supera
    ``muestra_maxima`` observaciones, se toma una muestra aleatoria con semilla
    fija para mantener la reproducibilidad y evitar gráficas ilegibles.
    """
    import numpy as np

    real = np.asarray(y_real, dtype=float).ravel()
    predicho = np.asarray(y_predicho, dtype=float).ravel()

    if real.size > muestra_maxima:
        generador = np.random.default_rng(42)
        indices = generador.choice(real.size, size=muestra_maxima, replace=False)
        real, predicho = real[indices], predicho[indices]

    plt = obtener_pyplot()
    figura, ejes = nueva_figura(
        titulo="Valores reales vs. predichos",
        etiqueta_x=f"{etiqueta_objetivo} real",
        etiqueta_y=f"{etiqueta_objetivo} predicho",
    )
    ejes.scatter(
        real, predicho, s=14, alpha=0.45, color="#2ca02c", edgecolors="none",
        label=f"Muestra de {real.size} observaciones",
    )

    inferior = min(real.min(), predicho.min())
    superior = max(real.max(), predicho.max())
    ejes.plot(
        [inferior, superior], [inferior, superior], color="#d62728",
        linewidth=2, linestyle="--", label="Predicción perfecta",
    )
    ejes.legend()

    ruta = exportar(figura, clave_modelo, nombre_archivo)
    cerrar(figura)
    return ruta


def grafica_residuos(
    y_real: Any,
    y_predicho: Any,
    clave_modelo: str,
    *,
    etiqueta_objetivo: str,
    nombre_archivo: str = "residuos.png",
) -> Path:
    """Distribución de los residuos y su dispersión frente a la predicción.

    Un buen ajuste lineal muestra residuos centrados en cero y sin patrón
    sistemático.
    """
    import numpy as np

    plt = obtener_pyplot()
    real = np.asarray(y_real, dtype=float).ravel()
    predicho = np.asarray(y_predicho, dtype=float).ravel()
    residuos = real - predicho

    figura, (ejes_hist, ejes_disp) = plt.subplots(1, 2, figsize=(13, 4.8))

    ejes_hist.hist(residuos, bins=40, color="#1f77b4", alpha=0.85, edgecolor="white")
    ejes_hist.axvline(0, color="#d62728", linestyle="--", linewidth=2, label="Residuo = 0")
    ejes_hist.set_title("Distribución de residuos")
    ejes_hist.set_xlabel(f"Residuo ({etiqueta_objetivo})")
    ejes_hist.set_ylabel("Frecuencia")
    ejes_hist.legend()

    ejes_disp.scatter(predicho, residuos, s=12, alpha=0.4, color="#9467bd", edgecolors="none")
    ejes_disp.axhline(0, color="#d62728", linestyle="--", linewidth=2)
    ejes_disp.set_title("Residuos vs. valores predichos")
    ejes_disp.set_xlabel(f"{etiqueta_objetivo} predicho")
    ejes_disp.set_ylabel("Residuo")

    figura.tight_layout()
    ruta = exportar(figura, clave_modelo, nombre_archivo)
    cerrar(figura)
    return ruta


def barras_coeficientes(
    coeficientes: Sequence[Any],
    clave_modelo: str,
    *,
    etiqueta_objetivo: str,
    nombre_archivo: str = "coeficientes_estandarizados.png",
    titulo: str = "Impacto de cada variable (β estandarizado)",
) -> Path:
    """Barras horizontales con los coeficientes estandarizados de cada variable.

    Se representan los β estandarizados porque son comparables entre variables
    medidas en escalas distintas.
    """
    import numpy as np

    plt = obtener_pyplot()
    nombres = [c.nombre for c in coeficientes]
    betas = [c.beta_estandarizado for c in coeficientes]
    colores = ["#2ca02c" if b >= 0 else "#d62728" for b in betas]

    figura, ejes = nueva_figura(titulo=titulo, etiqueta_x="β estandarizado", tamano=(8.5, 4.6))

    posiciones = np.arange(len(nombres))
    ejes.barh(posiciones, betas, color=colores, alpha=0.85)
    ejes.set_yticks(posiciones)
    ejes.set_yticklabels(nombres)
    ejes.axvline(0, color="black", linewidth=0.9)
    ejes.invert_yaxis()

    maximo = max((abs(b) for b in betas), default=0.0)
    desplazamiento = 0.02 * maximo if maximo else 0.01
    for posicion, beta in zip(posiciones, betas):
        ejes.text(
            beta + (desplazamiento if beta >= 0 else -desplazamiento),
            posicion,
            f"{beta:+.4f}",
            va="center",
            ha="left" if beta >= 0 else "right",
            fontsize=9,
        )

    ejes.set_xlabel(f"β estandarizado (efecto sobre {etiqueta_objetivo})")
    figura.tight_layout()
    ruta = exportar(figura, clave_modelo, nombre_archivo)
    cerrar(figura)
    return ruta


__all__ = [
    "TAMANO_PREDETERMINADO",
    "DPI_PREDETERMINADO",
    "obtener_pyplot",
    "configurar_estilo",
    "nueva_figura",
    "exportar",
    "cerrar",
    "dispersion_frente_objetivo",
    "mapa_calor_correlacion",
    "real_frente_predicho",
    "grafica_residuos",
    "barras_coeficientes",
]
