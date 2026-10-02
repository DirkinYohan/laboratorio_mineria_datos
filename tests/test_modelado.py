"""Pruebas del entrenamiento y de la inferencia estadística.

Verifican que la regresión lineal múltiple se ajuste correctamente, que la
partición sea reproducible y sin fuga de datos, y que la inferencia estadística
coincida con un cálculo de mínimos cuadrados ordinarios independiente.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common.metricas import calcular_metricas
from src.common.modelado import (
    PROPORCION_PRUEBA,
    SEMILLA,
    entrenar_regresion,
    estandarizar_impacto,
    factores_inflacion_varianza,
)
from src.common.paths import DATA_PROCESSED_DIR

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
    return (
        df.loc[:, list(especificacion.variables)],
        df[especificacion.objetivo],
    )


@pytest.fixture
def entrenado(clave: str, datos):
    """Modelo ya entrenado para cada modelo."""
    especificacion = especificacion_de(clave)
    X, y = datos
    return entrenar_regresion(
        X,
        y,
        variables=especificacion.variables,
        objetivo=especificacion.objetivo,
        funcion_metricas=calcular_metricas,
    )


# --- Partición de datos -------------------------------------------------
def test_la_semilla_es_fija() -> None:
    """La semilla declarada debe ser 42 para garantizar reproducibilidad."""
    assert SEMILLA == 42
    assert PROPORCION_PRUEBA == 0.2


def test_el_entrenamiento_es_reproducible(clave: str, datos) -> None:
    """Dos entrenamientos deben dar idéntico resultado."""
    especificacion = especificacion_de(clave)
    X, y = datos

    primero = entrenar_regresion(X, y, variables=especificacion.variables)
    segundo = entrenar_regresion(X, y, variables=especificacion.variables)

    assert primero.intercepto == pytest.approx(segundo.intercepto)
    for a, b in zip(primero.coeficientes, segundo.coeficientes):
        assert a.estimacion == pytest.approx(b.estimacion)


def test_proporcion_de_la_particion(clave: str, entrenado) -> None:
    """El conjunto de prueba debe ser aproximadamente el 20 %."""
    particion = entrenado.particion
    total = len(particion["y_entrenamiento"]) + len(particion["y_prueba"])
    assert len(particion["y_prueba"]) == pytest.approx(total * 0.2, abs=1)


def test_particiones_sin_fuga_de_datos(clave: str, entrenado) -> None:
    """Ninguna observación debe estar a la vez en entrenamiento y prueba."""
    particion = entrenado.particion
    indices_ent = set(particion["y_entrenamiento"].index)
    indices_prueba = set(particion["y_prueba"].index)

    assert not (indices_ent & indices_prueba), "Hay fuga de datos entre particiones"


# --- Coeficientes -------------------------------------------------------
def test_hay_un_coeficiente_por_variable(clave: str, entrenado) -> None:
    """Debe haber exactamente un coeficiente por variable independiente."""
    especificacion = especificacion_de(clave)
    assert len(entrenado.coeficientes) == len(especificacion.variables)
    assert tuple(c.nombre for c in entrenado.coeficientes) == especificacion.variables


def test_intercepto_coincide_con_el_estimador(clave: str, entrenado) -> None:
    """El intercepto reportado debe coincidir con el del estimador."""
    assert entrenado.intercepto == pytest.approx(entrenado.estimador.intercept_)


def test_coeficientes_coinciden_con_el_estimador(clave: str, entrenado) -> None:
    """Cada coeficiente reportado debe coincidir con el del estimador."""
    for indice, coeficiente in enumerate(entrenado.coeficientes):
        assert coeficiente.estimacion == pytest.approx(
            entrenado.estimador.coef_[indice]
        )


def test_errores_estandar_positivos(clave: str, entrenado) -> None:
    """Los errores estándar deben ser estrictamente positivos."""
    for coeficiente in entrenado.coeficientes:
        assert coeficiente.error_estandar > 0


def test_intervalo_de_confianza_contiene_la_estimacion(clave: str, entrenado) -> None:
    """El intervalo al 95 % debe contener la estimación puntual."""
    for coeficiente in entrenado.coeficientes:
        assert (
            coeficiente.intervalo.inferior
            < coeficiente.estimacion
            < coeficiente.intervalo.superior
        )


def test_p_valores_son_probabilidades(clave: str, entrenado) -> None:
    """Los p-valores deben estar en [0, 1]."""
    for coeficiente in entrenado.coeficientes:
        assert 0.0 <= coeficiente.valor_p <= 1.0


def test_inferencia_coincide_con_ols(clave: str, entrenado) -> None:
    """La inferencia debe coincidir con un cálculo OLS hecho desde cero.

    Se recalcula la matriz de covarianzas y se compara con lo reportado.
    """
    X = entrenado.particion["X_entrenamiento"].to_numpy(dtype=float)
    y = entrenado.particion["y_entrenamiento"].to_numpy(dtype=float)
    diseno = np.column_stack([np.ones(len(X)), X])

    beta, *_ = np.linalg.lstsq(diseno, y, rcond=None)
    residuos = y - diseno @ beta
    grados = len(y) - X.shape[1] - 1
    covarianza = (residuos @ residuos / grados) * np.linalg.inv(diseno.T @ diseno)
    errores = np.sqrt(np.diag(covarianza))

    assert entrenado.intercepto == pytest.approx(beta[0], abs=1e-6)
    for indice, coeficiente in enumerate(entrenado.coeficientes):
        assert coeficiente.estimacion == pytest.approx(beta[indice + 1], abs=1e-6)
        assert coeficiente.error_estandar == pytest.approx(
            errores[indice + 1], rel=1e-6
        )


# --- Beta estandarizado y VIF ------------------------------------------
def test_beta_estandarizado_usa_desviacion_por_columna() -> None:
    """β debe usar la desviación de cada columna, no una global.

    Protege contra un error real: usar la desviación global de la matriz
    produciría valores incorrectos y llevaría a concluir que la variable
    equivocada tiene el mayor impacto.
    """
    X = pd.DataFrame(
        {
            "pequena": np.array([0.0, 1.0, 2.0, 3.0]),
            "grande": np.array([1000.0, 2000.0, 3000.0, 4000.0]),
        }
    )
    betas = estandarizar_impacto(X, np.array([10.0, 10.0]), 1.0)

    assert betas[0] == pytest.approx(10.0 * X["pequena"].std(ddof=0))
    assert betas[1] == pytest.approx(10.0 * X["grande"].std(ddof=0))
    assert betas[1] > betas[0] * 100


def test_beta_estandarizado_rechaza_objetivo_constante() -> None:
    """Un objetivo con desviación cero debe provocar un error claro."""
    with pytest.raises(ValueError):
        estandarizar_impacto(pd.DataFrame({"a": [1.0, 2.0]}), np.array([1.0]), 0.0)


def test_vif_cercano_a_uno_sin_colinealidad() -> None:
    """Variables sin correlación deben tener VIF cercano a 1."""
    generador = np.random.default_rng(42)
    X = pd.DataFrame(generador.normal(size=(500, 3)), columns=["a", "b", "c"])
    assert np.all(factores_inflacion_varianza(X) < 1.1)


def test_vif_detecta_colinealidad() -> None:
    """Una variable duplicada debe producir un VIF muy alto."""
    generador = np.random.default_rng(42)
    base = generador.normal(size=300)
    X = pd.DataFrame({"a": base, "b": base, "c": generador.normal(size=300)})

    vifs = factores_inflacion_varianza(X)
    assert vifs[0] > 100 and vifs[1] > 100


def test_vif_de_una_sola_variable_es_uno() -> None:
    """Con una sola variable no hay multicolinealidad posible."""
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    assert factores_inflacion_varianza(X)[0] == pytest.approx(1.0)


def test_los_modelos_del_laboratorio_no_tienen_colinealidad(clave: str, entrenado) -> None:
    """Los tres modelos deben presentar VIF cercanos a 1."""
    for coeficiente in entrenado.coeficientes:
        assert coeficiente.vif < 5, f"VIF alto en '{coeficiente.nombre}'"


# --- Interpretación -----------------------------------------------------
def test_mayor_impacto_usa_beta_estandarizado(clave: str, entrenado) -> None:
    """La variable dominante debe ser la del mayor |β|."""
    esperado = max(entrenado.coeficientes, key=lambda c: abs(c.beta_estandarizado))
    assert entrenado.mas_influyente.nombre == esperado.nombre


def test_coeficientes_significativos_filtrados(clave: str, entrenado) -> None:
    """Los significativos deben tener p-valor menor que 0.05."""
    for coeficiente in entrenado.coeficientes_significativos:
        assert coeficiente.valor_p < 0.05


def test_la_ecuacion_incluye_intercepto_y_variables(clave: str, entrenado) -> None:
    """La ecuación debe incluir el intercepto y todas las variables."""
    ecuacion = entrenado.ecuacion()
    assert entrenado.objetivo in ecuacion
    for nombre in entrenado.variables:
        assert nombre in ecuacion


def test_tabla_de_coeficientes_completa(clave: str, entrenado) -> None:
    """La tabla debe incluir la inferencia completa."""
    tabla = entrenado.tabla_coeficientes()
    esperadas = {
        "variable", "coeficiente", "error_estandar", "t", "p_valor",
        "beta_estandarizado", "VIF", "ic_95_inferior", "ic_95_superior",
        "significativo",
    }
    assert esperadas.issubset(set(tabla.columns))
    assert len(tabla) == 3


def test_el_resumen_es_serializable(clave: str, entrenado) -> None:
    """El resumen debe poder convertirse a JSON."""
    import json

    payload = entrenado.resumen()
    json.dumps(payload)

    assert payload["modelo"] == especificacion_de(clave).objetivo
    assert "variable_mayor_impacto" in payload


# --- Calidad del ajuste -------------------------------------------------
def test_r2_es_interpretable(clave: str, entrenado) -> None:
    """El R² debe estar entre 0 y 1."""
    assert 0.0 <= entrenado.metricas_prueba["r2"] <= 1.0


def test_los_modelos_superan_a_la_media(clave: str, entrenado) -> None:
    """Cada modelo debe explicar varianza (R² > 0)."""
    assert entrenado.metricas_prueba["r2"] > 0.0
    assert entrenado.metricas_entrenamiento["r2"] > 0.0
