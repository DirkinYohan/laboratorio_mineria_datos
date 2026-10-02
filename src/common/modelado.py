"""Entrenamiento de regresión lineal múltiple (genérico y reutilizable).

Contiene la lógica de *modelado* del proyecto. No conoce detalles particulares
de dólar, glucosa o energía: recibe la matriz de variables y el vector objetivo.

Además del ajuste con ``scikit-learn``, calcula la **inferencia estadística**
del modelo (errores estándar, estadísticos t, p-valores), los **coeficientes
estandarizados** (β) y el **factor de inflación de varianza** (VIF). Estas
medidas son las que permiten interpretar los coeficientes y determinar con
criterio técnico qué variable tiene mayor impacto.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Semilla y proporción de la partición. Fijas para garantizar reproducibilidad.
SEMILLA: int = 42
PROPORCION_PRUEBA: float = 0.2


@dataclass(frozen=True)
class Intervalo:
    """Intervalo de confianza de un coeficiente."""

    inferior: float
    superior: float


@dataclass(frozen=True)
class Coeficiente:
    """Coeficiente estimado con su inferencia estadística."""

    nombre: str
    estimacion: float
    error_estandar: float
    estadistico_t: float
    valor_p: float
    beta_estandarizado: float
    vif: float
    intervalo: Intervalo

    @property
    def significativo(self) -> bool:
        """Indica si el coeficiente es significativo al 5 %."""
        return self.valor_p < 0.05

    @property
    def etiqueta_significancia(self) -> str:
        """Descripción legible de la significancia estadística."""
        if self.valor_p < 0.001:
            return "altamente significativo (p < 0,001)"
        if self.valor_p < 0.01:
            return "muy significativo (p < 0,01)"
        if self.valor_p < 0.05:
            return "significativo (p < 0,05)"
        return "no significativo (p >= 0,05)"


@dataclass
class ModeloEntrenado:
    """Resultado del entrenamiento y de la evaluación estadística.

    Attributes:
        estimador: Modelo de regresión ajustado (scikit-learn).
        variables: Nombres de las variables independientes, en orden.
        objetivo: Nombre de la variable objetivo.
        intercepto: Intercepto del modelo.
        coeficientes: Coeficientes con inferencia estadística.
        particion: Conjunto de datos de entrenamiento y prueba.
        metricas_entrenamiento: Métricas sobre el conjunto de entrenamiento.
        metricas_prueba: Métricas sobre el conjunto de prueba.
        nota_metodologica: Comentario sobre el ajuste.

    Nota:
        Los coeficientes y su inferencia se calculan sobre el conjunto de
        **entrenamiento**, que es la muestra con la que se ajustó el estimador.
        Las métricas de desempeño se reportan sobre el conjunto de **prueba**,
        que el modelo no vio durante el ajuste.
    """

    estimador: Any
    variables: tuple[str, ...]
    objetivo: str
    intercepto: float
    coeficientes: tuple[Coeficiente, ...]
    particion: dict[str, Any] = field(repr=False)
    metricas_entrenamiento: dict[str, float] = field(default_factory=dict)
    metricas_prueba: dict[str, float] = field(default_factory=dict)
    nota_metodologica: str = ""

    def predecir(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Genera predicciones para nuevas observaciones."""
        return self.estimador.predict(X)

    @property
    def mas_influyente(self) -> Coeficiente:
        """Variable con mayor impacto según el valor absoluto de β estandarizado.

        Se usa β estandarizado y no el coeficiente bruto porque los coeficientes
        brutos no son comparables entre variables medidas en escalas distintas.
        """
        return max(self.coeficientes, key=lambda c: abs(c.beta_estandarizado))

    @property
    def coeficientes_significativos(self) -> tuple[Coeficiente, ...]:
        """Coeficientes significativos al 5 %."""
        return tuple(c for c in self.coeficientes if c.significativo)

    def tabla_coeficientes(self) -> pd.DataFrame:
        """Tabla de coeficientes lista para exportar y para el informe."""
        return pd.DataFrame(
            [
                {
                    "variable": c.nombre,
                    "coeficiente": c.estimacion,
                    "error_estandar": c.error_estandar,
                    "t": c.estadistico_t,
                    "p_valor": c.valor_p,
                    "beta_estandarizado": c.beta_estandarizado,
                    "VIF": c.vif,
                    "ic_95_inferior": c.intervalo.inferior,
                    "ic_95_superior": c.intervalo.superior,
                    "significativo": c.significativo,
                }
                for c in self.coeficientes
            ]
        )

    def ecuacion(self) -> str:
        """Ecuación de regresión estimada, lista para el informe."""
        terminos = " ".join(
            f"{'+' if c.estimacion >= 0 else '-'} {abs(c.estimacion):.4f}·{c.nombre}"
            for c in self.coeficientes
        )
        return f"{self.objetivo} = {self.intercepto:.4f} {terminos}"

    def resumen(self) -> dict[str, Any]:
        """Resumen completo del modelo en formato serializable."""
        return {
            "modelo": self.objetivo,
            "variables": list(self.variables),
            "intercepto": self.intercepto,
            "ecuacion": self.ecuacion(),
            "coeficientes": [
                {
                    "variable": c.nombre,
                    "coeficiente": c.estimacion,
                    "error_estandar": c.error_estandar,
                    "t": c.estadistico_t,
                    "p_valor": c.valor_p,
                    "beta_estandarizado": c.beta_estandarizado,
                    "VIF": c.vif,
                    "ic_95": [c.intervalo.inferior, c.intervalo.superior],
                    "significativo": c.significativo,
                    "interpretacion_significancia": c.etiqueta_significancia,
                }
                for c in self.coeficientes
            ],
            "variable_mayor_impacto": {
                "segun_beta_estandarizado": self.mas_influyente.nombre,
                "beta_estandarizado": self.mas_influyente.beta_estandarizado,
            },
            "metricas_entrenamiento": self.metricas_entrenamiento,
            "metricas_prueba": self.metricas_prueba,
            "nota_metodologica": self.nota_metodologica,
        }


def estandarizar_impacto(
    X: pd.DataFrame, coeficientes: np.ndarray, desviacion_objetivo: float
) -> np.ndarray:
    """Calcula los coeficientes estandarizados (β).

    β_j = b_j · (s_xj / s_y), donde ``s_xj`` es la desviación estándar de la
    variable ``j`` y ``s_y`` la de la variable objetivo.

    Es imprescindible calcular ``s_xj`` **por columna**: usar una desviación
    global produciría un valor incorrecto y llevaría a conclusiones erróneas
    sobre qué variable tiene mayor impacto.

    Raises:
        ValueError: Si la desviación del objetivo es cero.
    """
    if desviacion_objetivo == 0:
        raise ValueError("La desviación estándar del objetivo es cero.")

    desviaciones = np.asarray(X.std(axis=0, ddof=0), dtype=float).ravel()
    return np.asarray(coeficientes, dtype=float).ravel() * desviaciones / desviacion_objetivo


def factores_inflacion_varianza(X: pd.DataFrame) -> np.ndarray:
    """Calcula el VIF de cada variable independiente.

    Un VIF cercano a 1 indica ausencia de multicolinealidad; por encima de 10
    suele considerarse problemático.
    """
    valores = np.asarray(X, dtype=float)
    n_columnas = valores.shape[1]
    vifs = np.ones(n_columnas, dtype=float)

    if n_columnas < 2:
        return vifs

    for j in range(n_columnas):
        otras = np.delete(valores, j, axis=1)
        modelo = LinearRegression().fit(otras, valores[:, j])
        r2 = modelo.score(otras, valores[:, j])
        vifs[j] = np.inf if r2 >= 1.0 else 1.0 / (1.0 - r2)

    return vifs


def _coeficientes_con_inferencia(
    X: pd.DataFrame, y: pd.Series, estimador: LinearRegression
) -> tuple[Coeficiente, ...]:
    """Estima la inferencia estadística de los coeficientes por OLS.

    Reproduce el ajuste con álgebra lineal para obtener la matriz de covarianzas
    de los estimadores y, con ella, errores estándar, estadísticos t, p-valores e
    intervalos de confianza al 95 %.

    IMPORTANTE: ``X`` e ``y`` deben ser **la misma muestra** con la que se ajustó
    ``estimador``. Calcular los errores estándar sobre otra muestra invalidaría
    los estadísticos t y los p-valores.

    Raises:
        ValueError: Si no hay grados de libertad suficientes.
    """
    valores_x = np.asarray(X, dtype=float)
    valores_y = np.asarray(y, dtype=float).ravel()
    n_muestras, n_variables = valores_x.shape

    # Matriz aumentada con la columna de unos (intercepto).
    diseno = np.column_stack([np.ones(n_muestras), valores_x])
    beta, *_ = np.linalg.lstsq(diseno, valores_y, rcond=None)
    residuos = valores_y - diseno @ beta

    grados_libertad = n_muestras - n_variables - 1
    if grados_libertad <= 0:
        raise ValueError(
            "No hay grados de libertad suficientes para la inferencia estadística."
        )

    varianza_residual = float(residuos @ residuos) / grados_libertad
    covarianza = varianza_residual * np.linalg.inv(diseno.T @ diseno)
    errores = np.sqrt(np.diag(covarianza))
    estadisticos_t = beta / errores
    valores_p = 2.0 * (1.0 - stats.t.cdf(np.abs(estadisticos_t), grados_libertad))
    critico = stats.t.ppf(0.975, grados_libertad)

    betas = estandarizar_impacto(X, estimador.coef_, float(np.std(valores_y, ddof=0)))
    vifs = factores_inflacion_varianza(X)

    coeficientes: list[Coeficiente] = []
    for indice, nombre in enumerate(X.columns):
        posicion = indice + 1  # +1 porque beta[0] es el intercepto.
        coeficientes.append(
            Coeficiente(
                nombre=str(nombre),
                estimacion=float(estimador.coef_[indice]),
                error_estandar=float(errores[posicion]),
                estadistico_t=float(estadisticos_t[posicion]),
                valor_p=float(valores_p[posicion]),
                beta_estandarizado=float(betas[indice]),
                vif=float(vifs[indice]),
                intervalo=Intervalo(
                    inferior=float(beta[posicion] - critico * errores[posicion]),
                    superior=float(beta[posicion] + critico * errores[posicion]),
                ),
            )
        )
    return tuple(coeficientes)


def dividir_datos(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    proporcion_prueba: float = PROPORCION_PRUEBA,
    semilla: int = SEMILLA,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divide los datos en conjuntos de entrenamiento y prueba.

    La semilla es fija para garantizar resultados reproducibles, requisito
    habitual en un trabajo académico.
    """
    return train_test_split(X, y, test_size=proporcion_prueba, random_state=semilla)


def entrenar_regresion(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    variables: tuple[str, ...] | None = None,
    objetivo: str = "objetivo",
    proporcion_prueba: float = PROPORCION_PRUEBA,
    semilla: int = SEMILLA,
    funcion_metricas: Callable[[Any, Any], dict[str, float]] | None = None,
    nota_metodologica: str = "",
) -> ModeloEntrenado:
    """Entrena un modelo de regresión lineal múltiple y calcula su inferencia.

    Args:
        X: Matriz de variables independientes (sin nulos, numérica).
        y: Vector de la variable objetivo.
        variables: Nombres de las variables; si se omite se toman de ``X``.
        objetivo: Nombre de la variable objetivo.
        proporcion_prueba: Proporción reservada para prueba.
        semilla: Semilla para reproducibilidad.
        funcion_metricas: Función ``(y_real, y_pred) -> dict`` para las métricas.
        nota_metodologica: Comentario a incluir en el resumen.

    Returns:
        Un ``ModeloEntrenado`` con el estimador, la inferencia y las métricas.
    """
    nombres = tuple(variables) if variables is not None else tuple(X.columns)
    X = X.loc[:, list(nombres)]

    X_ent, X_prueba, y_ent, y_prueba = dividir_datos(
        X, y, proporcion_prueba=proporcion_prueba, semilla=semilla
    )

    estimador = LinearRegression()
    estimador.fit(X_ent, y_ent)

    # La inferencia se calcula sobre la MISMA muestra del ajuste (entrenamiento):
    # usar otra muestra invalidaría los estadísticos t y los p-valores.
    coeficientes = _coeficientes_con_inferencia(X_ent, y_ent, estimador)

    metricas_ent: dict[str, float] = {}
    metricas_prueba: dict[str, float] = {}
    if funcion_metricas is not None:
        metricas_ent = funcion_metricas(y_ent, estimador.predict(X_ent))
        metricas_prueba = funcion_metricas(y_prueba, estimador.predict(X_prueba))

    return ModeloEntrenado(
        estimador=estimador,
        variables=nombres,
        objetivo=objetivo,
        intercepto=float(estimador.intercept_),
        coeficientes=coeficientes,
        particion={
            "X_entrenamiento": X_ent,
            "X_prueba": X_prueba,
            "y_entrenamiento": y_ent,
            "y_prueba": y_prueba,
        },
        metricas_entrenamiento=metricas_ent,
        metricas_prueba=metricas_prueba,
        nota_metodologica=nota_metodologica,
    )


__all__ = [
    "SEMILLA",
    "PROPORCION_PRUEBA",
    "Intervalo",
    "Coeficiente",
    "ModeloEntrenado",
    "estandarizar_impacto",
    "factores_inflacion_varianza",
    "dividir_datos",
    "entrenar_regresion",
]
