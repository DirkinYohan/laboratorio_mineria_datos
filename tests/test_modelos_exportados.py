"""Pruebas de la exportación, carga y consumo de los modelos.

Verifican que cada artefacto ``.joblib`` pueda cargarse de nuevo, que realice
predicciones correctas con el número y orden de variables esperados, y que los
resultados guardados sean coherentes.

Estas pruebas definen el **contrato de integración** que consumirá el frontend de
Next.js: cargar el ``.joblib`` y pasarle exactamente las variables declaradas.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.common import io
from src.common.paths import DATA_PROCESSED_DIR, MODELS_DIR, RESULTS_DIR

CLAVES = ("dolar", "glucosa", "energia")

# Métricas exigidas por el laboratorio para cada modelo.
METRICAS_EXIGIDAS = {
    "dolar": ("mse", "r2"),
    "glucosa": ("mse", "r2"),
    "energia": ("rmse", "r2"),
}


def especificacion_de(clave: str):
    """Especificación de cada modelo."""
    modulo = __import__(f"src.{clave}.especificacion", fromlist=["ESPECIFICACION"])
    return modulo.ESPECIFICACION


def modulo(clave: str, nombre: str):
    """Importa un módulo de un modelo."""
    return __import__(f"src.{clave}.{nombre}", fromlist=[nombre])


@pytest.fixture(params=CLAVES)
def clave(request) -> str:
    """Clave de cada modelo."""
    return request.param


@pytest.fixture
def especificacion(clave: str):
    """Especificación de cada modelo."""
    return especificacion_de(clave)


@pytest.fixture
def fila_real(especificacion) -> dict[str, float]:
    """Una fila real del dataset procesado, como diccionario de entrada."""
    df = pd.read_csv(DATA_PROCESSED_DIR / especificacion.archivo_procesado)
    fila = df.iloc[0]
    return {nombre: float(fila[nombre]) for nombre in especificacion.variables}


# --- Artefactos exportados ---------------------------------------------
@pytest.mark.parametrize("clave", CLAVES)
def test_el_modelo_exportado_existe(clave: str) -> None:
    """Cada modelo debe estar exportado como ``.joblib``."""
    especificacion = especificacion_de(clave)
    assert especificacion.ruta_modelo.is_file(), (
        f"Falta el modelo exportado '{especificacion.ruta_modelo}'"
    )


def test_los_tres_modelos_existen() -> None:
    """Deben existir exactamente los tres modelos exportados."""
    estado = {
        clave: (MODELS_DIR / clave / f"modelo_{clave}.joblib").is_file()
        for clave in CLAVES
    }
    assert estado == {"dolar": True, "glucosa": True, "energia": True}


def test_el_artefacto_esta_en_su_carpeta(especificacion, clave: str) -> None:
    """El artefacto debe ubicarse en ``models/<clave>/``."""
    assert especificacion.ruta_modelo.parent == MODELS_DIR / clave


# --- Carga y predicción -------------------------------------------------
def test_el_modelo_se_puede_recargar(especificacion) -> None:
    """El artefacto debe poder cargarse de nuevo desde disco."""
    estimador = io.cargar_modelo(especificacion)

    assert hasattr(estimador, "predict")
    assert hasattr(estimador, "coef_")
    assert hasattr(estimador, "intercept_")


def test_el_modelo_espera_tres_variables(especificacion) -> None:
    """El modelo debe esperar exactamente tres variables."""
    estimador = io.cargar_modelo(especificacion)
    assert estimador.n_features_in_ == len(especificacion.variables) == 3


def test_el_modelo_predice_para_datos_reales(clave: str, fila_real) -> None:
    """Debe devolver un número para una fila real del dataset."""
    especificacion = especificacion_de(clave)
    estimador = io.cargar_modelo(especificacion)

    cuadro = pd.DataFrame(
        [[fila_real[n] for n in especificacion.variables]],
        columns=list(especificacion.variables),
    )
    prediccion = float(estimador.predict(cuadro)[0])

    assert prediccion == prediccion  # No es NaN.
    assert isinstance(prediccion, float)


def test_la_prediccion_es_coherente_con_el_objetivo(clave: str, fila_real) -> None:
    """La predicción debe ser del orden de magnitud del objetivo real."""
    especificacion = especificacion_de(clave)
    estimador = io.cargar_modelo(especificacion)
    df = pd.read_csv(DATA_PROCESSED_DIR / especificacion.archivo_procesado)

    cuadro = pd.DataFrame(
        [[fila_real[n] for n in especificacion.variables]],
        columns=list(especificacion.variables),
    )
    prediccion = float(estimador.predict(cuadro)[0])

    minimo, maximo = df[especificacion.objetivo].min(), df[especificacion.objetivo].max()
    margen = 0.5 * (maximo - minimo)
    assert minimo - margen <= prediccion <= maximo + margen


def test_predecir_rechaza_variables_faltantes(clave: str) -> None:
    """El módulo de evaluación debe rechazar matrices incompletas."""
    evaluacion = modulo(clave, "evaluacion")
    especificacion = especificacion_de(clave)
    estimador = io.cargar_modelo(especificacion)

    incompleto = pd.DataFrame([[1.0]], columns=[especificacion.variables[0]])

    with pytest.raises(ValueError):
        evaluacion.predecir(estimador, incompleto)


def test_el_orden_de_las_variables_importa(clave: str, fila_real) -> None:
    """Invertir el orden de las columnas debe cambiar la predicción.

    Documenta el contrato de integración: el frontend debe enviar las variables
    en el orden declarado por la especificación.
    """
    evaluacion = modulo(clave, "evaluacion")
    especificacion = especificacion_de(clave)
    estimador = io.cargar_modelo(especificacion)

    orden_correcto = pd.DataFrame(
        [[fila_real[n] for n in especificacion.variables]],
        columns=list(especificacion.variables),
    )
    orden_invertido = pd.DataFrame(
        [[fila_real[n] for n in reversed(especificacion.variables)]],
        columns=list(reversed(especificacion.variables)),
    )

    pred_correcta = evaluacion.predecir(estimador, orden_correcto)
    pred_por_nombre = evaluacion.predecir(estimador, orden_invertido)

    # Al seleccionar por nombre, el resultado debe ser el mismo.
    assert pred_correcta == pytest.approx(pred_por_nombre)


# --- Resultados guardados ----------------------------------------------
@pytest.mark.parametrize("clave", CLAVES)
def test_los_resultados_son_json_valido(clave: str) -> None:
    """Los resultados guardados deben ser JSON válido y estar completos."""
    carpeta = RESULTS_DIR / clave
    for nombre in (
        "metricas.json", "modelo.json", "interpretacion.json", "analisis.json",
        "limpieza.json", "auditoria_cruda.json", "comprension_datos.json",
        "residuos.json",
    ):
        ruta = carpeta / nombre
        assert ruta.is_file(), f"Falta '{ruta}'"
        json.loads(ruta.read_text(encoding="utf-8"))


def test_las_metricas_exigidas_estan_presentes(clave: str) -> None:
    """Cada modelo debe tener las métricas que exige el laboratorio.

    Dólar y Glucosa: MSE y R². Energía: RMSE y R².
    """
    metricas = json.loads(
        (RESULTS_DIR / clave / "metricas.json").read_text(encoding="utf-8")
    )["metricas"]

    for exigida in METRICAS_EXIGIDAS[clave]:
        assert exigida in metricas, f"Falta la métrica '{exigida}' en {clave}"


def test_las_metricas_son_coherentes(clave: str) -> None:
    """Las métricas guardadas deben ser coherentes entre sí."""
    metricas = json.loads(
        (RESULTS_DIR / clave / "metricas.json").read_text(encoding="utf-8")
    )["metricas"]

    assert 0.0 <= metricas["r2"] <= 1.0
    assert metricas["rmse"] == pytest.approx(metricas["mse"] ** 0.5)
    assert metricas["mae"] > 0


def test_la_tabla_de_coeficientes_tiene_tres_filas(clave: str) -> None:
    """Debe haber una fila por variable independiente."""
    especificacion = especificacion_de(clave)
    tabla = pd.read_csv(RESULTS_DIR / clave / "coeficientes.csv")

    assert len(tabla) == len(especificacion.variables) == 3
    assert set(tabla["variable"]) == set(especificacion.variables)


def test_el_analisis_incluye_intercepto_y_coeficientes(clave: str) -> None:
    """El análisis debe exponer intercepto y coeficientes."""
    analisis = json.loads(
        (RESULTS_DIR / clave / "analisis.json").read_text(encoding="utf-8")
    )
    especificacion = especificacion_de(clave)

    assert "intercepto" in analisis
    assert isinstance(analisis["intercepto"], (int, float))
    assert set(analisis["coeficientes"]) == set(especificacion.variables)
    assert analisis["variable_objetivo"] == especificacion.objetivo


def test_las_predicciones_guardadas_tienen_residuos(clave: str) -> None:
    """La tabla de predicciones debe incluir residuo y error absoluto."""
    tabla = pd.read_csv(RESULTS_DIR / clave / "predicciones.csv")

    assert list(tabla.columns) == ["real", "predicho", "residuo", "error_absoluto"]
    assert len(tabla) > 0
    assert (tabla["error_absoluto"] >= 0).all()


def test_la_interpretacion_incluye_ranking(clave: str) -> None:
    """La interpretación debe incluir el ranking de impacto."""
    interpretacion = json.loads(
        (RESULTS_DIR / clave / "interpretacion.json").read_text(encoding="utf-8")
    )

    assert "variable_mayor_impacto" in interpretacion
    assert len(interpretacion["ranking_impacto"]) == 3
    assert len(interpretacion["interpretaciones"]) == 3


# --- Coherencia con el pipeline ----------------------------------------
def test_la_evaluacion_del_pipeline_coincide_con_la_guardada(clave: str) -> None:
    """Evaluar de nuevo debe reproducir exactamente las métricas guardadas."""
    evaluacion = modulo(clave, "evaluacion")
    guardadas = json.loads(
        (RESULTS_DIR / clave / "metricas.json").read_text(encoding="utf-8")
    )["metricas"]

    nuevas = evaluacion.evaluar()["metricas"]

    assert nuevas["r2"] == pytest.approx(guardadas["r2"])
    assert nuevas["mse"] == pytest.approx(guardadas["mse"])
    assert nuevas["rmse"] == pytest.approx(guardadas["rmse"])


def test_el_intercepto_coincide_con_el_modelo(clave: str) -> None:
    """El intercepto del análisis debe coincidir con el del ``.joblib``."""
    especificacion = especificacion_de(clave)
    evaluacion = modulo(clave, "evaluacion")
    estimador = io.cargar_modelo(especificacion)

    ajuste = evaluacion.obtener_coeficientes(estimador)
    assert ajuste["intercepto"] == pytest.approx(float(estimador.intercept_))
    assert ajuste["coeficientes"][especificacion.variables[0]] == pytest.approx(
        float(estimador.coef_[0])
    )
