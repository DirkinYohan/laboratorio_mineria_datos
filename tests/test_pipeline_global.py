"""Pruebas del orquestador global (``src.pipeline``).

Verifican que exista un único comando capaz de construir los tres modelos, que
el registro de modelos sea coherente con las especificaciones declaradas y que el
resumen final detecte correctamente los artefactos ausentes.

No se entrena ningún modelo aquí: eso lo hace el pipeline real. Estas pruebas
comprueban la orquestación, no el aprendizaje.
"""

from __future__ import annotations

import pytest

from src import pipeline
from src.common import io
from src.common.paths import MODELS_DIR


# --- Registro de modelos -------------------------------------------------
def test_registro_cubre_los_tres_modelos() -> None:
    """El orquestador debe declarar los tres modelos del laboratorio."""
    assert pipeline.CLAVES == ("dolar", "glucosa", "energia")


def test_registro_no_tiene_claves_repetidas() -> None:
    """Cada modelo debe aparecer una sola vez."""
    assert len(pipeline.CLAVES) == len(set(pipeline.CLAVES))


@pytest.mark.parametrize("clave", pipeline.CLAVES)
def test_cada_modelo_declara_sus_modulos(clave: str) -> None:
    """Cada entrada debe apuntar a un orquestador y a una especificación."""
    modelo = pipeline.obtener(clave)
    assert modelo.orquestador == f"src.{clave}.pipeline"
    assert modelo.especificacion == f"src.{clave}.especificacion"


@pytest.mark.parametrize("clave", pipeline.CLAVES)
def test_los_modulos_declarados_existen(clave: str) -> None:
    """Los módulos declarados en el registro deben ser importables."""
    import importlib

    modelo = pipeline.obtener(clave)
    assert importlib.import_module(modelo.orquestador) is not None
    assert importlib.import_module(modelo.especificacion) is not None


def test_los_pipelines_exponen_la_misma_interfaz() -> None:
    """Todos los pipelines deben ofrecer las mismas funciones públicas."""
    from src.dolar import pipeline as referencia

    firma = {nombre for nombre in dir(referencia) if not nombre.startswith("_")}

    for clave in pipeline.CLAVES:
        modulo = __import__(f"src.{clave}.pipeline", fromlist=["pipeline"])
        assert "main" in dir(modulo), f"src.{clave}.pipeline no expone 'main'"
        assert "ejecutar" in dir(modulo), f"src.{clave}.pipeline no expone 'ejecutar'"
        assert firma <= set(dir(modulo)), (
            f"src.{clave}.pipeline no replica la interfaz de referencia"
        )


def test_modelo_desconocido_falla() -> None:
    """Pedir un modelo inexistente debe fallar de forma explícita."""
    with pytest.raises(KeyError):
        pipeline.obtener("inexistente")


# --- Etapas admitidas ----------------------------------------------------
def test_etapas_cubren_las_de_los_pipelines() -> None:
    """Las etapas del orquestador deben coincidir con las de cada modelo."""
    assert set(pipeline.ETAPAS) == {
        "auditar",
        "limpiar",
        "modelar",
        "evaluar",
        "graficar",
        "completo",
    }


# --- Verificación de artefactos -----------------------------------------
@pytest.mark.parametrize("clave", pipeline.CLAVES)
def test_verificar_detecta_artefacto_ausente(clave: str, monkeypatch) -> None:
    """Si el ``.joblib`` no existe, la verificación debe avisar."""
    def cargar_fallido(especificacion):
        raise FileNotFoundError(f"No se encontró el modelo exportado de {clave}.")

    monkeypatch.setattr(io, "cargar_modelo", cargar_fallido)

    with pytest.raises(FileNotFoundError):
        pipeline.verificar(clave)


@pytest.mark.parametrize("clave", pipeline.CLAVES)
def test_verificar_rechaza_modelo_con_variables_incompletas(
    clave: str, monkeypatch
) -> None:
    """Un artefacto con un número de variables distinto al declarado es inválido."""
    class EstimadorFalso:
        n_features_in_ = 99
        coef_ = [0.0] * 99
        intercept_ = 0.0

    monkeypatch.setattr(io, "cargar_modelo", lambda especificacion: EstimadorFalso())

    with pytest.raises(ValueError, match="variables"):
        pipeline.verificar(clave)


@pytest.mark.parametrize("clave", pipeline.CLAVES)
def test_verificar_artefacto_real(clave: str) -> None:
    """El artefacto de cada modelo debe cargar y tener 3 variables."""
    especificacion = pipeline._especificacion(clave)
    if not especificacion.ruta_modelo.is_file():
        pytest.skip(f"modelo_{clave}.joblib no está construido")

    estado = pipeline.verificar(clave)
    assert estado["clave"] == clave
    assert estado["variables"] == len(especificacion.variables)
    assert len(estado["coeficientes"]) == len(especificacion.variables)


def test_rutas_de_modelos_apuntan_a_models() -> None:
    """Los tres artefactos se ubican bajo ``models/<clave>/``."""
    for clave in pipeline.CLAVES:
        especificacion = pipeline._especificacion(clave)
        assert especificacion.ruta_modelo.parent == MODELS_DIR / clave
