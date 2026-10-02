"""Pruebas de la estructura del proyecto y de los contratos de datos.

Verifican que la arquitectura sea coherente, que **no exista ningún componente
de frontend** y que los datasets crudos correspondan a las especificaciones
declaradas.
"""

from __future__ import annotations

import csv

import pytest

from src.common.paths import (
    DATA_RAW_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    VISUALIZATIONS_DIR,
)

# Especificaciones esperadas por el enunciado del laboratorio.
ESPECIFICACIONES = {
    "dolar": {
        "variables": ("Dia", "Inflacion", "Tasa_interes"),
        "objetivo": "Precio_Dolar",
        "crudo": "dolar_data.csv",
        "procesado": "dolar_clean.csv",
        "modelo": "modelo_dolar.joblib",
    },
    "glucosa": {
        "variables": ("Edad", "IMC", "Actividad_Fisica"),
        "objetivo": "Nivel_Glucosa",
        "crudo": "glucosa_data.csv",
        "procesado": "glucosa_clean.csv",
        "modelo": "modelo_glucosa.joblib",
    },
    "energia": {
        "variables": ("Temperatura", "Hora", "Dia_Semana"),
        "objetivo": "Consumo_Energia",
        "crudo": "energia_data.csv",
        "procesado": "energia_clean.csv",
        "modelo": "modelo_energia.joblib",
    },
}

CLAVES = sorted(ESPECIFICACIONES)

# Los cuatro módulos que debe exponer cada paquete de modelo.
MODULOS_POR_MODELO = ("limpieza", "modelo", "evaluacion", "visualizacion")


@pytest.fixture(params=CLAVES)
def clave(request) -> str:
    """Clave de cada modelo del laboratorio."""
    return request.param


@pytest.fixture
def especificacion(clave: str):
    """Especificación del modelo indicado."""
    from src.common.especificacion import EspecificacionModelo

    modulo = __import__(f"src.{clave}.especificacion", fromlist=["ESPECIFICACION"])
    assert isinstance(modulo.ESPECIFICACION, EspecificacionModelo)
    return modulo.ESPECIFICACION


# --- Capas del proyecto -------------------------------------------------
@pytest.mark.parametrize(
    "capa",
    ["data", "src", "models", "visualizations", "results", "tests", "informe"],
)
def test_capas_existen(capa: str) -> None:
    """Todas las capas del proyecto deben existir."""
    assert (PROJECT_ROOT / capa).is_dir(), f"Falta la carpeta '{capa}/'"


def test_archivos_raiz_existen() -> None:
    """El proyecto debe incluir sus archivos de configuración de la raíz."""
    for nombre in ("requirements.txt", "README.md", ".gitignore", "src", "tests"):
        assert (PROJECT_ROOT / nombre).exists(), f"Falta '{nombre}'"


@pytest.mark.parametrize("capa", ["src", "models", "results", "visualizations"])
def test_subcarpetas_por_modelo_existen(capa: str, clave: str) -> None:
    """Cada modelo debe tener su carpeta propia en cada capa."""
    assert (PROJECT_ROOT / capa / clave).is_dir()


@pytest.mark.parametrize("modulo", MODULOS_POR_MODELO)
def test_modulos_por_modelo_existen(modulo: str, clave: str) -> None:
    """Cada paquete de modelo debe exponer los cuatro módulos requeridos."""
    ruta = PROJECT_ROOT / "src" / clave / f"{modulo}.py"
    assert ruta.is_file(), f"Falta 'src/{clave}/{modulo}.py'"


@pytest.mark.parametrize(
    "modulo",
    ["especificacion", "paths", "limpieza", "io", "modelado", "metricas",
     "visualizacion"],
)
def test_modulos_comunes_existen(modulo: str) -> None:
    """La capa compartida debe contener sus módulos núcleo."""
    assert (PROJECT_ROOT / "src" / "common" / f"{modulo}.py").is_file()


# --- Ausencia de frontend ----------------------------------------------
def test_no_existe_carpeta_app() -> None:
    """No debe existir ningún paquete de interfaz web."""
    assert not (PROJECT_ROOT / "app").exists(), (
        "La carpeta 'app/' no debe existir: el frontend se desarrolla aparte."
    )


@pytest.mark.parametrize(
    "archivo",
    ["app/main.py", "app/model_registry.py", "main.py", "streamlit_app.py"],
)
def test_no_existen_archivos_de_interfaz(archivo: str) -> None:
    """No debe existir ningún archivo de interfaz web."""
    assert not (PROJECT_ROOT / archivo).exists(), f"No debe existir '{archivo}'"


@pytest.mark.parametrize(
    "termino",
    ["streamlit", "flask", "fastapi", "django", "uvicorn", "gradio", "dash"],
)
def test_requirements_no_declara_dependencias_web(termino: str) -> None:
    """``requirements.txt`` no debe declarar dependencias de interfaz web."""
    contenido = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    declaradas = [
        linea.strip()
        for linea in contenido.splitlines()
        if linea.strip() and not linea.strip().startswith("#")
    ]
    infractoras = [linea for linea in declaradas if linea.lower().startswith(termino)]
    assert not infractoras, f"Dependencias web declaradas: {infractoras}"


def test_codigo_fuente_no_importa_librerias_web() -> None:
    """Ningún módulo de ``src`` debe importar librerías de interfaz."""
    librerias = ("streamlit", "flask", "fastapi", "django", "gradio", "dash")
    infractores: list[str] = []

    for ruta in (PROJECT_ROOT / "src").rglob("*.py"):
        texto = ruta.read_text(encoding="utf-8")
        for libreria in librerias:
            if f"import {libreria}" in texto:
                infractores.append(f"{ruta.name}: import {libreria}")

    assert not infractores, f"Código con dependencias de interfaz: {infractores}"


# --- Especificaciones ---------------------------------------------------
def test_especificacion_declara_variables(especificacion, clave: str) -> None:
    """Cada modelo debe declarar las variables del enunciado, en orden."""
    esperado = ESPECIFICACIONES[clave]
    assert especificacion.clave == clave
    assert especificacion.variables == esperado["variables"]
    assert especificacion.objetivo == esperado["objetivo"]


def test_especificacion_declara_archivos(especificacion, clave: str) -> None:
    """Cada modelo debe declarar sus tres archivos (crudo, procesado, modelo)."""
    esperado = ESPECIFICACIONES[clave]
    assert especificacion.archivo_crudo == esperado["crudo"]
    assert especificacion.archivo_procesado == esperado["procesado"]
    assert especificacion.archivo_modelo == esperado["modelo"]


def test_especificacion_cubre_todas_las_variables(especificacion) -> None:
    """Cada variable independiente debe tener metadatos declarados."""
    nombres = tuple(v.nombre for v in especificacion.especificaciones)
    assert nombres == especificacion.variables


def test_ruta_del_modelo_es_correcta(especificacion, clave: str) -> None:
    """El artefacto debe ubicarse en ``models/<clave>/``."""
    assert especificacion.ruta_modelo.parent == MODELS_DIR / clave
    assert especificacion.ruta_modelo.suffix == ".joblib"


# --- Datasets crudos ----------------------------------------------------
def test_dataset_crudo_existe(especificacion) -> None:
    """El CSV crudo debe existir en ``data/raw``."""
    assert especificacion.ruta_cruda.is_file(), f"Falta '{especificacion.ruta_cruda}'"


def test_cabecera_del_crudo_coincide(especificacion) -> None:
    """El CSV crudo debe tener las columnas declaradas, en orden."""
    with especificacion.ruta_cruda.open(newline="", encoding="utf-8") as manejador:
        cabecera = next(csv.reader(manejador))
    assert tuple(cabecera) == especificacion.columnas


def test_dataset_crudo_no_esta_vacio(especificacion) -> None:
    """El dataset crudo debe contener registros, no solo la cabecera."""
    with especificacion.ruta_cruda.open(newline="", encoding="utf-8") as manejador:
        assert sum(1 for _ in manejador) > 1


def test_los_tres_modelos_tienen_subcarpeta_de_resultados(clave: str) -> None:
    """Los resultados se organizan por modelo."""
    assert (RESULTS_DIR / clave).parent == RESULTS_DIR
    assert (VISUALIZATIONS_DIR / clave).parent == VISUALIZATIONS_DIR
    assert DATA_RAW_DIR.is_dir()
