"""Pruebas de la limpieza y preparación de los datos.

Comprueban que la limpieza sea reproducible, que **nunca modifique los datos
crudos** y que los datasets procesados cumplan las condiciones necesarias para el
modelado.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.common import limpieza as comun
from src.common.paths import DATA_PROCESSED_DIR, DATA_RAW_DIR

CLAVES = ("dolar", "glucosa", "energia")

# Columnas de cada modelo, para no depender de importaciones dinámicas.
COLUMNAS = {
    "dolar": ("Dia", "Inflacion", "Tasa_interes", "Precio_Dolar"),
    "glucosa": ("Edad", "IMC", "Actividad_Fisica", "Nivel_Glucosa"),
    "energia": ("Temperatura", "Hora", "Dia_Semana", "Consumo_Energia"),
}


def modulo_limpieza(clave: str):
    """Importa el módulo de limpieza del modelo indicado."""
    return __import__(f"src.{clave}.limpieza", fromlist=["limpieza"])


@pytest.fixture(params=CLAVES)
def clave(request) -> str:
    """Clave de cada modelo."""
    return request.param


@pytest.fixture
def modulo(clave: str):
    """Módulo de limpieza de cada modelo."""
    return modulo_limpieza(clave)


@pytest.fixture
def especificacion(clave: str):
    """Especificación de cada modelo."""
    modulo = __import__(f"src.{clave}.especificacion", fromlist=["ESPECIFICACION"])
    return modulo.ESPECIFICACION


# Ejemplos de filas válidas y con defectos para cada modelo, de modo que cada
# prueba use datos coherentes con la especificación que recibe.
EJEMPLO_VALIDO = {
    "dolar": {
        "Dia": [1, 2, 3],
        "Inflacion": [0.0224, 0.0193, 0.0232],
        "Tasa_interes": [5.46, 5.95, 4.30],
        "Precio_Dolar": [4024.83, 4000.54, 3979.62],
    },
    "glucosa": {
        "Edad": [42, 67, 71],
        "IMC": [22.5, 23.4, 25.7],
        "Actividad_Fisica": [1, 3, 8],
        "Nivel_Glucosa": [141.84, 171.93, 155.50],
    },
    "energia": {
        "Temperatura": [20.0, 25.0, 30.0],
        "Hora": [1, 12, 24],
        "Dia_Semana": [1, 4, 7],
        "Consumo_Energia": [300.0, 400.0, 500.0],
    },
}


# --- Inmutabilidad de los datos crudos ---------------------------------
def test_el_pipeline_no_modifica_los_crudos(modulo, especificacion) -> None:
    """Ejecutar la preparación no debe alterar el CSV de ``data/raw``."""
    antes = especificacion.ruta_cruda.read_bytes()

    modulo.preparar(persistir=True)

    despues = especificacion.ruta_cruda.read_bytes()
    assert antes == despues, "Se modificó un dataset original"


def test_los_crudos_conservan_sus_columnas(especificacion) -> None:
    """Los CSV crudos deben mantener las columnas declaradas."""
    crudo = pd.read_csv(especificacion.ruta_cruda)
    assert list(crudo.columns) == list(especificacion.columnas)
    assert DATA_RAW_DIR in especificacion.ruta_cruda.parents


# --- Dataset procesado --------------------------------------------------
def test_existe_el_dataset_procesado(especificacion) -> None:
    """Cada modelo debe tener su dataset procesado."""
    ruta = DATA_PROCESSED_DIR / especificacion.archivo_procesado
    assert ruta.is_file(), f"Falta '{ruta}'. Ejecuta la limpieza del modelo."


def test_el_procesado_solo_tiene_las_columnas_del_modelo(especificacion) -> None:
    """El dataset procesado debe contener exactamente las columnas del modelo."""
    procesado = pd.read_csv(DATA_PROCESSED_DIR / especificacion.archivo_procesado)
    assert list(procesado.columns) == list(especificacion.columnas)


def test_el_procesado_no_tiene_nulos(especificacion) -> None:
    """No debe haber valores nulos tras la limpieza."""
    procesado = pd.read_csv(DATA_PROCESSED_DIR / especificacion.archivo_procesado)
    assert int(procesado.isna().sum().sum()) == 0


def test_el_procesado_no_tiene_duplicados(especificacion) -> None:
    """No debe haber filas duplicadas tras la limpieza."""
    procesado = pd.read_csv(DATA_PROCESSED_DIR / especificacion.archivo_procesado)
    assert int(procesado.duplicated().sum()) == 0


def test_el_procesado_esta_en_rangos_plausibles(especificacion) -> None:
    """Todos los valores deben estar dentro de los rangos de dominio."""
    procesado = pd.read_csv(DATA_PROCESSED_DIR / especificacion.archivo_procesado)
    infractoras = comun.comprobar_rangos(procesado, especificacion)
    assert infractoras == {}, f"Valores fuera de rango: {infractoras}"


def test_el_procesado_es_numerico(especificacion) -> None:
    """Todas las columnas deben ser numéricas."""
    procesado = pd.read_csv(DATA_PROCESSED_DIR / especificacion.archivo_procesado)
    for columna in procesado.columns:
        assert pd.api.types.is_numeric_dtype(procesado[columna]), (
            f"'{columna}' no es numérica"
        )


def test_variables_enteras_son_enteras(especificacion) -> None:
    """Las variables marcadas como enteras deben venir como enteros."""
    procesado = pd.read_csv(DATA_PROCESSED_DIR / especificacion.archivo_procesado)
    for variable in especificacion.especificaciones:
        if variable.entera:
            assert pd.api.types.is_integer_dtype(procesado[variable.nombre]), (
                f"'{variable.nombre}' debería ser entero"
            )


def test_no_se_pierden_filas_sin_defectos(especificacion) -> None:
    """Los datasets del laboratorio no tienen defectos: no se pierde ninguna fila."""
    crudo = pd.read_csv(especificacion.ruta_cruda)
    procesado = pd.read_csv(DATA_PROCESSED_DIR / especificacion.archivo_procesado)
    assert len(procesado) == len(crudo)


def test_los_crudos_no_tienen_nulos_ni_duplicados(especificacion) -> None:
    """Verifica la premisa anterior sobre los datos reales."""
    crudo = pd.read_csv(especificacion.ruta_cruda)
    assert int(crudo.isna().sum().sum()) == 0
    assert len(comun.detectar_duplicados(crudo)) == 0


# --- Funciones de limpieza ---------------------------------------------
def test_deteccion_de_atipicos_es_consistente(especificacion) -> None:
    """El criterio IQR debe marcar exactamente lo que cae fuera de sus límites."""
    crudo = pd.read_csv(especificacion.ruta_cruda)
    serie = crudo[especificacion.objetivo]
    mascara = comun.detectar_atipicos_iqr(serie)

    q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
    rango = q3 - q1
    esperado = (serie < q1 - 1.5 * rango) | (serie > q3 + 1.5 * rango)
    assert mascara.equals(esperado)


def test_los_atipicos_plausibles_se_conservan(clave: str, especificacion) -> None:
    """Los atípicos plausibles no deben eliminarse (decisión documentada).

    Se comprueba en todas las columnas del modelo. ``Precio_Dolar``, por ejemplo,
    no presenta atípicos IQR en su objetivo, así que la aserción se apoya en
    cuántos hay realmente y no en un supuesto sobre el dataset.
    """
    crudo = pd.read_csv(especificacion.ruta_cruda)
    procesado = pd.read_csv(DATA_PROCESSED_DIR / especificacion.archivo_procesado)

    total_atipicos = sum(
        int(comun.detectar_atipicos_iqr(crudo[columna]).sum())
        for columna in especificacion.columnas
    )

    # El dataset del laboratorio sí tiene atípicos (en inflación, tasa, IMC,
    # glucosa, temperatura o consumo), y ninguno debe haberse eliminado.
    assert total_atipicos > 0, "Se esperaban atípicos en alguna variable del modelo"
    assert len(procesado) == len(crudo), "Los atípicos plausibles no deben eliminarse"


def test_eliminar_filas_invalidas_quita_nulos_y_duplicados(clave: str, especificacion) -> None:
    """Debe eliminar nulos y duplicados, y reportarlo."""
    base = EJEMPLO_VALIDO[clave]
    columnas = list(base)

    primera = {nombre: valores[0] for nombre, valores in base.items()}
    duplicada = dict(primera)
    con_nulo = {nombre: valores[1] for nombre, valores in base.items()}
    con_nulo[columnas[0]] = None
    tercera = {nombre: valores[2] for nombre, valores in base.items()}

    cuadro = pd.DataFrame([primera, duplicada, con_nulo, tercera], columns=columnas)
    limpio, reporte = comun.eliminar_filas_invalidas(cuadro, especificacion)

    assert reporte["nulos"] == 1
    assert reporte["duplicados"] == 1
    assert len(limpio) == 2
    assert int(limpio.isna().sum().sum()) == 0
    assert int(limpio.duplicated().sum()) == 0


def test_eliminar_filas_invalidas_quita_valores_imposibles(
    clave: str, especificacion
) -> None:
    """Los valores fuera del rango plausible deben eliminarse."""
    base = EJEMPLO_VALIDO[clave]
    columnas = list(base)
    variable_objetivo = especificacion.objetivo

    correcta = {nombre: valores[0] for nombre, valores in base.items()}
    correcta2 = {nombre: valores[1] for nombre, valores in base.items()}

    # Se fuerza un valor imposible en la variable objetivo de este modelo.
    imposible = {nombre: valores[2] for nombre, valores in base.items()}
    imposible[variable_objetivo] = especificacion.especificacion_objetivo.maximo + 1000

    cuadro = pd.DataFrame([correcta, correcta2, imposible], columns=columnas)
    limpio, reporte = comun.eliminar_filas_invalidas(cuadro, especificacion)

    assert reporte["fuera_de_rango"] == 1
    assert len(limpio) == 2
    assert limpio[variable_objetivo].max() <= especificacion.especificacion_objetivo.maximo


def test_ajustar_tipos_no_convierte_proporciones_en_enteros(
    clave: str, especificacion
) -> None:
    """Las variables continuas pequenas no deben redondearse a entero.

    Protege contra un error real: deducir el tipo del rango habría convertido
    ``Inflacion`` (0-1) en 0 o 1, perdiendo la información.
    """
    cuadro = pd.DataFrame(EJEMPLO_VALIDO[clave], columns=list(EJEMPLO_VALIDO[clave]))
    resultado = comun.ajustar_tipos(cuadro, especificacion)

    # Cada variable declarada como entera debe seguir siendo entera.
    for variable in especificacion.especificaciones:
        if variable.entera:
            assert pd.api.types.is_integer_dtype(resultado[variable.nombre]), (
                f"'{variable.nombre}' debería ser entero"
            )

    # Y ninguna variable continua debe haber perdido precisión.
    for variable in especificacion.especificaciones:
        if not variable.entera:
            original = cuadro[variable.nombre].tolist()
            assert resultado[variable.nombre].tolist() == pytest.approx(original), (
                f"'{variable.nombre}' perdió precisión al ajustar los tipos"
            )


def test_dolar_no_convierte_la_inflacion_en_enteros() -> None:
    """Caso concreto del error evitado: ``Inflacion`` va de 0 a 1."""
    modulo = __import__("src.dolar.especificacion", fromlist=["ESPECIFICACION"])
    inflacion = modulo.ESPECIFICACION.especificacion_de("Inflacion")

    assert inflacion.entera is False, (
        "'Inflacion' no es un recuento entero: es una proporción continua"
    )

    cuadro = pd.DataFrame(EJEMPLO_VALIDO["dolar"])
    resultado = comun.ajustar_tipos(cuadro, modulo.ESPECIFICACION)

    assert pd.api.types.is_float_dtype(resultado["Inflacion"])
    assert resultado["Inflacion"].tolist() == pytest.approx([0.0224, 0.0193, 0.0232])


def test_resumen_de_limpieza_cuenta_filas(especificacion) -> None:
    """El reporte de limpieza debe reflejar las filas eliminadas."""
    reporte = comun.resumen_limpieza(
        especificacion, filas_antes=100, eliminaciones={"nulos": 5}, filas_despues=95
    )
    assert reporte["filas_antes"] == 100
    assert reporte["filas_despues"] == 95
    assert reporte["filas_eliminadas"] == 5
    assert "criterio_atipicos" in reporte


def test_guardar_procesado_no_toca_los_crudos(tmp_path, monkeypatch, especificacion) -> None:
    """Guardar un dataset procesado no debe afectar a ``data/raw``."""
    monkeypatch.setattr(comun, "DATA_PROCESSED_DIR", tmp_path)

    procesado = pd.read_csv(DATA_PROCESSED_DIR / especificacion.archivo_procesado)
    antes = especificacion.ruta_cruda.read_bytes()

    escrito = comun.guardar_procesado(procesado.head(10), especificacion)

    assert escrito.is_file()
    assert escrito.parent == tmp_path
    assert especificacion.ruta_cruda.read_bytes() == antes


# --- Preparación por modelo --------------------------------------------
def test_separar_variables_devuelve_X_e_y(modulo, especificacion) -> None:
    """Debe separar correctamente las variables independientes del objetivo."""
    X, y = modulo.preparar(persistir=False)

    assert list(X.columns) == list(especificacion.variables)
    assert y.name == especificacion.objetivo
    assert len(X) == len(y)
    assert X.shape[1] == 3
