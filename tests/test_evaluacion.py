"""Pruebas de las métricas de evaluación y de las visualizaciones.

Verifican que las métricas sean correctas (comparadas con cálculos manuales),
que la línea base se comporte como corresponde y que las gráficas se generen sin
errores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common import visualizacion as viz
from src.common.metricas import (
    calcular_metricas,
    evaluar_modelo,
    interpretar_coeficiente,
    interpretar_desempeno,
    metricas_linea_base,
    r2_ajustado,
    resumen_residuos,
    tabla_predicciones,
)
from src.common.modelado import entrenar_regresion
from src.common.paths import DATA_PROCESSED_DIR, VISUALIZATIONS_DIR

CLAVES = ("dolar", "glucosa", "energia")


def especificacion_de(clave: str):
    """Especificación de cada modelo."""
    modulo = __import__(f"src.{clave}.especificacion", fromlist=["ESPECIFICACION"])
    return modulo.ESPECIFICACION


@pytest.fixture(params=CLAVES)
def clave(request) -> str:
    """Clave de cada modelo."""
    return request.param


@pytest.fixture
def datos(clave: str):
    """Datos procesados ``(X, y)`` de cada modelo."""
    especificacion = especificacion_de(clave)
    df = pd.read_csv(DATA_PROCESSED_DIR / especificacion.archivo_procesado)
    return df.loc[:, list(especificacion.variables)], df[especificacion.objetivo]


@pytest.fixture
def entrenado(clave: str, datos):
    """Modelo entrenado de cada modelo."""
    especificacion = especificacion_de(clave)
    X, y = datos
    return entrenar_regresion(
        X, y, variables=especificacion.variables, objetivo=especificacion.objetivo,
        funcion_metricas=calcular_metricas,
    )


# --- Métricas -----------------------------------------------------------
def test_metricas_con_prediccion_perfecta() -> None:
    """Una predicción perfecta debe dar R²=1 y errores nulos."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    metricas = calcular_metricas(y, y)

    assert metricas["r2"] == pytest.approx(1.0)
    assert metricas["mse"] == pytest.approx(0.0)
    assert metricas["rmse"] == pytest.approx(0.0)


def test_metricas_coinciden_con_calculo_manual() -> None:
    """Las métricas deben coincidir con el cálculo a mano."""
    real = np.array([10.0, 20.0, 30.0])
    predicho = np.array([12.0, 18.0, 33.0])
    metricas = calcular_metricas(real, predicho)

    errores = real - predicho
    assert metricas["mse"] == pytest.approx(float(np.mean(errores**2)))
    assert metricas["mae"] == pytest.approx(float(np.mean(np.abs(errores))))


def test_rmse_es_la_raiz_del_mse() -> None:
    """RMSE debe ser exactamente la raíz del MSE."""
    generador = np.random.default_rng(0)
    real = generador.normal(size=100)
    predicho = real + generador.normal(scale=0.5, size=100)
    metricas = calcular_metricas(real, predicho)

    assert metricas["rmse"] == pytest.approx(np.sqrt(metricas["mse"]))


def test_r2_ajustado_penaliza() -> None:
    """El R² ajustado debe ser menor o igual que el R²."""
    assert r2_ajustado(0.8, n_muestras=100, n_variables=3) < 0.8


def test_r2_ajustado_exige_grados_de_libertad() -> None:
    """Sin grados de libertad suficientes debe lanzar un error claro."""
    with pytest.raises(ValueError):
        r2_ajustado(0.9, n_muestras=4, n_variables=3)


def test_linea_base_predice_la_media() -> None:
    """La línea base debe equivaler a predecir la media del entrenamiento."""
    y_entrenamiento = np.array([10.0, 20.0, 30.0])
    y_prueba = np.array([10.0, 20.0, 30.0])
    base = metricas_linea_base(y_entrenamiento, y_prueba)

    esperado = calcular_metricas(y_prueba, np.full(3, 20.0))
    assert base["rmse"] == pytest.approx(esperado["rmse"])
    assert base["r2"] == 0.0


def test_evaluar_modelo_reporta_mejora(clave: str, entrenado) -> None:
    """El modelo debe mejorar a la línea base y reportarlo."""
    resultado = evaluar_modelo(entrenado)

    assert set(resultado) == {"metricas", "linea_base", "mejora_rmse_vs_linea_base"}
    assert resultado["metricas"]["r2_ajustado"] <= resultado["metricas"]["r2"]
    assert resultado["mejora_rmse_vs_linea_base"] > 0


def test_tabla_de_predicciones_incluye_residuos() -> None:
    """La tabla debe incluir el residuo y el error absoluto."""
    tabla = tabla_predicciones(np.array([1.0, 2.0, 3.0]), np.array([1.5, 2.0, 2.5]))

    assert list(tabla.columns) == ["real", "predicho", "residuo", "error_absoluto"]
    assert tabla["residuo"].tolist() == pytest.approx([-0.5, 0.0, 0.5])


def test_los_residuos_de_ols_suman_cero(clave: str) -> None:
    """En una regresión con intercepto, la media de los residuos es cero."""
    generador = np.random.default_rng(1)
    x = generador.normal(size=200)
    y = 3.0 + 2.0 * x + generador.normal(scale=0.1, size=200)

    modelo = entrenar_regresion(pd.DataFrame({"x": x}), pd.Series(y), variables=("x",))
    residuo = np.asarray(modelo.particion["y_entrenamiento"]) - modelo.predecir(
        modelo.particion["X_entrenamiento"]
    )
    assert float(np.mean(residuo)) == pytest.approx(0.0, abs=1e-9)


def test_resumen_de_residuos_completo() -> None:
    """El resumen debe incluir dispersión y forma de la distribución."""
    resumen = resumen_residuos(
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    )
    assert set(resumen) == {
        "residuo_medio", "residuo_std", "residuo_min", "residuo_max",
        "asimetria", "curtosis",
    }


def test_interpretacion_del_coeficiente_refleja_el_signo() -> None:
    """La interpretación debe reflejar el signo del coeficiente."""
    positivo = interpretar_coeficiente("Edad", 1.5, "años", "Glucosa", "mg/dL")
    negativo = interpretar_coeficiente("Actividad", -2.0, "nivel", "Glucosa", "mg/dL")

    assert "aumenta" in positivo and "1.5000" in positivo
    assert "disminuye" in negativo and "2.0000" in negativo


def test_interpretacion_del_desempeno_usa_las_metricas(clave: str, entrenado) -> None:
    """El texto debe derivarse de las métricas reales del modelo."""
    metricas = entrenado.metricas_prueba
    texto = interpretar_desempeno(metricas, "la variable", "unidad")

    assert f"{metricas['r2']:.4f}" in texto
    assert f"{metricas['mae']:.4f}" in texto


# --- Visualizaciones ---------------------------------------------------
def test_dispersion_se_genera(clave: str, datos, tmp_path, monkeypatch) -> None:
    """La dispersión de una variable frente al objetivo debe exportarse."""
    monkeypatch.setattr(viz.io, "VISUALIZATIONS_DIR", tmp_path)
    especificacion = especificacion_de(clave)
    X, y = datos
    df = X.copy()
    df[especificacion.objetivo] = y

    ruta = viz.dispersion_frente_objetivo(
        df, especificacion.variables[0], especificacion.objetivo, clave,
        etiqueta_variable="Variable", etiqueta_objetivo="Objetivo",
        nombre_archivo="prueba_dispersion.png",
    )
    assert ruta.is_file() and ruta.stat().st_size > 0


def test_mapa_de_calor_se_genera(clave: str, datos, tmp_path, monkeypatch) -> None:
    """La matriz de correlación debe exportarse."""
    monkeypatch.setattr(viz.io, "VISUALIZATIONS_DIR", tmp_path)
    especificacion = especificacion_de(clave)
    X, y = datos
    df = X.copy()
    df[especificacion.objetivo] = y

    ruta = viz.mapa_calor_correlacion(
        df, especificacion.columnas, clave, nombre_archivo="prueba_correlacion.png"
    )
    assert ruta.is_file() and ruta.stat().st_size > 0


def test_real_frente_predicho_se_genera(clave: str, entrenado, tmp_path, monkeypatch) -> None:
    """La gráfica de reales frente a predichos debe exportarse."""
    monkeypatch.setattr(viz.io, "VISUALIZATIONS_DIR", tmp_path)
    particion = entrenado.particion

    ruta = viz.real_frente_predicho(
        particion["y_prueba"], entrenado.predecir(particion["X_prueba"]), clave,
        etiqueta_objetivo="Objetivo", nombre_archivo="prueba_real.png",
    )
    assert ruta.is_file()


def test_grafica_de_residuos_se_genera(clave: str, entrenado, tmp_path, monkeypatch) -> None:
    """La gráfica de residuos debe exportarse."""
    monkeypatch.setattr(viz.io, "VISUALIZATIONS_DIR", tmp_path)
    particion = entrenado.particion

    ruta = viz.grafica_residuos(
        particion["y_prueba"], entrenado.predecir(particion["X_prueba"]), clave,
        etiqueta_objetivo="Objetivo", nombre_archivo="prueba_residuos.png",
    )
    assert ruta.is_file()


def test_barras_de_coeficientes_se_generan(clave: str, entrenado, tmp_path, monkeypatch) -> None:
    """La gráfica de coeficientes estandarizados debe exportarse."""
    monkeypatch.setattr(viz.io, "VISUALIZATIONS_DIR", tmp_path)

    ruta = viz.barras_coeficientes(
        entrenado.coeficientes, clave, etiqueta_objetivo="Objetivo",
        nombre_archivo="prueba_coeficientes.png",
    )
    assert ruta.is_file()


def test_las_graficas_no_modifican_los_datos(clave: str, datos, tmp_path, monkeypatch) -> None:
    """Generar gráficas no debe alterar el dataset de entrada."""
    monkeypatch.setattr(viz.io, "VISUALIZATIONS_DIR", tmp_path)
    especificacion = especificacion_de(clave)
    X, y = datos
    df = X.copy()
    df[especificacion.objetivo] = y
    antes = df.copy(deep=True)

    viz.dispersion_frente_objetivo(
        df, especificacion.variables[0], especificacion.objetivo, clave,
        etiqueta_variable="Variable", etiqueta_objetivo="Objetivo",
        nombre_archivo="prueba_inmutabilidad.png",
    )
    pd.testing.assert_frame_equal(df, antes)


@pytest.mark.parametrize("clave", CLAVES)
def test_las_graficas_del_pipeline_existen(clave: str) -> None:
    """Cada modelo debe tener sus gráficas principales exportadas."""
    carpeta = VISUALIZATIONS_DIR / clave
    assert carpeta.is_dir()

    generadas = {p.name for p in carpeta.glob("*.png")}
    requeridas = {
        "matriz_correlacion.png",
        "real_vs_predicho.png",
        "residuos.png",
        "coeficientes_estandarizados.png",
    }
    faltantes = requeridas - generadas
    assert not faltantes, f"Gráficas faltantes en {clave}: {sorted(faltantes)}"
