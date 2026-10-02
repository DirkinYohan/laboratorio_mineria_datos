# Laboratorio de Minería de Datos — Regresión Lineal Múltiple con CRISP-DM

Proyecto **exclusivamente de Machine Learning**. Aplica la metodología
**CRISP-DM** para construir, evaluar y exportar tres modelos independientes de
**regresión lineal múltiple** con Python.

> **Este proyecto no incluye frontend.** La interfaz web se desarrollará por
> separado en **Next.js**, consumiendo los modelos `.joblib` exportados. Aquí no
> hay Streamlit, Flask, FastAPI, HTML, CSS ni JavaScript.

---

## Modelos

| Modelo | Variables independientes | Variable objetivo | Artefacto |
|--------|--------------------------|-------------------|-----------|
| Precio del Dólar | `Dia`, `Inflacion`, `Tasa_interes` | `Precio_Dolar` | `models/dolar/modelo_dolar.joblib` |
| Nivel de Glucosa | `Edad`, `IMC`, `Actividad_Fisica` | `Nivel_Glucosa` | `models/glucosa/modelo_glucosa.joblib` |
| Consumo de Energía | `Temperatura`, `Hora`, `Dia_Semana` | `Consumo_Energia` | `models/energia/modelo_energia.joblib` |

---

## Resultados obtenidos

Métricas sobre el **conjunto de prueba** (20 % de las observaciones, semilla fija 42).
Es la medida honesta de la capacidad de generalización, porque el modelo no vio
esas observaciones al ajustarse:

| Modelo | Métricas exigidas | R² | R² ajustado | MAE | MSE | RMSE | Mejora vs. media |
|--------|-------------------|------|-------------|-------|-------|-------|------------------|
| Precio del Dólar | MSE, R² | 0.9963 | 0.9962 | 37.01 | 2376.97 | 48.75 | 93.94 % |
| Nivel de Glucosa | MSE, R² | 0.6814 | 0.6790 | 12.22 | 233.69 | 15.29 | 43.59 % |
| Consumo de Energía | RMSE, R² | 0.8968 | 0.8967 | 16.51 | 429.52 | 20.72 | 67.88 % |

También se guardan las métricas sobre el **dataset completo**, que describen el
ajuste final sobre todos los datos disponibles (R² = 0.9951 / 0.6853 / 0.9005).
Ambos conjuntos están en `results/<modelo>/metricas.json`.

**Variable de mayor impacto** (según β estandarizado) y significancia:

| Modelo | Variable dominante | Efecto | Significativas (α = 0.05) |
|--------|-------------------|--------|---------------------------|
| Precio del Dólar | `Dia` | + | 1 de 3 |
| Nivel de Glucosa | `Edad` | + | 3 de 3 |
| Consumo de Energía | `Temperatura` | + | 3 de 3 |

El informe completo, con la interpretación de cada coeficiente y el análisis de
residuos, está en **[`informe/informe.docx`](informe/informe.docx)**. Todos los
valores que cita provienen de los artefactos de `results/`.

---

## Instalación

Requiere **Python 3.11 o superior**.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### Dependencias

| Paquete | Uso |
|---------|-----|
| `pandas` | Manipulación de datos y lectura de CSV |
| `numpy` | Operaciones numéricas |
| `scikit-learn` | Regresión lineal, partición de datos y métricas |
| `scipy` | Distribución t para p-valores e intervalos de confianza |
| `matplotlib` | Gráficas de análisis |
| `seaborn` | Mapa de calor de correlaciones |
| `joblib` | Exportación e importación de modelos |
| `pytest` | Pruebas automatizadas |

---

## Ejecución desde cero

```bash
# 1. Construir los TRES modelos con un solo comando
python -m src.pipeline

# 2. Ejecutar las pruebas automatizadas
pytest tests
```

Ese único comando recorre los tres modelos en orden (dólar → glucosa → energía) y,
al terminar, **comprueba que cada `.joblib` se haya generado y se pueda cargar**,
mostrando un resumen con sus métricas. Genera, para cada modelo: la auditoría de
calidad, el dataset limpio, el modelo entrenado, las métricas, la interpretación
de coeficientes y todas las gráficas.

Si algo falla en un modelo, los demás continúan y el comando termina con un
código de salida distinto de cero, indicando cuál falló y por qué.

### Opciones del comando global

```bash
python -m src.pipeline --etapa modelar        # solo entrena y exporta los tres
python -m src.pipeline --etapa auditar        # solo comprensión de datos
python -m src.pipeline --solo dolar           # únicamente el modelo del dólar
python -m src.pipeline --solo dolar --solo glucosa   # dos modelos concretos
```

| Opción | Qué hace |
|--------|----------|
| `--etapa` | Fase a ejecutar en cada modelo: `auditar`, `limpiar`, `modelar`, `evaluar`, `graficar` o `completo` (por defecto) |
| `--solo` | Limita la ejecución a uno o varios modelos. Se puede repetir |

### Ejecutar un modelo por separado

Cada modelo mantiene su propio punto de entrada, con las mismas etapas:

```bash
python -m src.dolar.pipeline                  # solo el modelo del dólar
python -m src.glucosa.pipeline --etapa evaluar
python -m src.energia.pipeline --etapa graficar
```

### Comandos por modelo

Solo estos módulos tienen punto de entrada propio. La limpieza no se ejecuta
sola: forma parte del pipeline y se invoca con `--etapa limpiar` (o
`--etapa limpiar` en el comando global para los tres a la vez).

```bash
python -m src.dolar.modelo         # solo entrenamiento y exportación
python -m src.dolar.evaluacion     # solo evaluación (carga el .joblib)
python -m src.dolar.visualizacion  # solo gráficas
python -m src.dolar.pipeline       # flujo completo del dólar
```

### Ejecución por fases

```bash
python -m src.dolar.pipeline --etapa auditar    # comprensión de los datos
python -m src.dolar.pipeline --etapa limpiar    # limpieza y data/processed
python -m src.dolar.pipeline --etapa modelar    # entrenamiento + evaluación
python -m src.dolar.pipeline --etapa evaluar    # solo evaluación
python -m src.dolar.pipeline --etapa graficar   # solo gráficas
```

---

## Estructura del proyecto

```text
laboratorio_mineria_datos/
│
├── data/
│   ├── raw/                      # Datasets originales (SOLO LECTURA)
│   │   ├── dolar_data.csv           500 filas
│   │   ├── glucosa_data.csv        2000 filas
│   │   └── energia_data.csv       10000 filas
│   └── processed/                # Datasets limpios (generados por el código)
│       ├── dolar_clean.csv
│       ├── glucosa_clean.csv
│       └── energia_clean.csv
│
├── src/
│   ├── pipeline.py               # Orquestador global: construye los tres modelos
│   ├── common/                   # Lógica compartida por los tres modelos
│   │   ├── especificacion.py     # Contrato de datos (variables, rangos, rutas)
│   │   ├── paths.py              # Rutas canónicas del proyecto
│   │   ├── limpieza.py           # Auditoría y limpieza reproducible
│   │   ├── io.py                 # Lectura/escritura de datos, modelos y figuras
│   │   ├── modelado.py           # Regresión lineal + inferencia estadística
│   │   ├── metricas.py           # Métricas, residuos e interpretación
│   │   └── visualizacion.py      # Estilo y gráficas genéricas
│   │
│   ├── dolar/
│   │   ├── especificacion.py     # Variables y rangos del modelo
│   │   ├── limpieza.py           # Limpieza y preparación
│   │   ├── modelo.py             # Entrenamiento y exportación
│   │   ├── evaluacion.py         # Métricas, coeficientes y análisis
│   │   ├── visualizacion.py      # Gráficas del modelo
│   │   └── pipeline.py           # Orquestador del flujo
│   │
│   ├── glucosa/                  # (misma estructura que dolar)
│   └── energia/                  # (misma estructura que dolar)
│
├── models/{dolar,glucosa,energia}/
│   └── modelo_<modelo>.joblib    # Modelos exportados
│
├── visualizations/{dolar,glucosa,energia}/
│   └── *.png                     # Gráficas de análisis (27 en total)
│
├── results/{dolar,glucosa,energia}/
│   ├── metricas.json             # MSE/RMSE, R², R² ajustado, MAE
│   ├── modelo.json               # Ecuación, coeficientes e inferencia
│   ├── interpretacion.json       # Interpretación y ranking de impacto
│   ├── analisis.json             # Análisis del desempeño y coeficientes
│   ├── auditoria_cruda.json      # Perfil de calidad de los datos crudos
│   ├── limpieza.json             # Registro de la limpieza aplicada
│   ├── comprension_datos.json    # Resumen de la fase de comprensión
│   ├── coeficientes.csv          # Tabla de coeficientes
│   ├── predicciones.csv          # Reales vs. predichos con residuos
│   └── residuos.json             # Estadísticos de los residuos
│
├── tests/                        # Suite de pruebas automatizadas
│   ├── conftest.py
│   ├── test_estructura.py        # Arquitectura, contratos y ausencia de frontend
│   ├── test_limpieza.py          # Limpieza y preparación
│   ├── test_modelado.py          # Entrenamiento e inferencia estadística
│   ├── test_evaluacion.py        # Métricas y visualizaciones
│   └── test_modelos_exportados.py # Exportación, carga y contrato de integración
│
├── informe/
│   └── informe.docx              # Informe completo del proyecto
│
├── requirements.txt
├── pytest.ini
├── README.md
└── .gitignore
```

### Principio de diseño

Cada modelo declara sus variables en `especificacion.py` y reutiliza los cuatro
módulos comunes. **No hay código duplicado ni nombres de columna repetidos** por
el código: la especificación es la única fuente de verdad.

```text
src/<modelo>/
├── especificacion.py   # Única declaración de variables, rangos y archivos
├── limpieza.py         # Reutiliza src/common/limpieza.py
├── modelo.py           # Reutiliza src/common/modelado.py
├── evaluacion.py       # Reutiliza src/common/metricas.py
└── visualizacion.py    # Reutiliza src/common/visualizacion.py
```

---

## Contrato de integración (para el frontend en Next.js)

El frontend consumirá los modelos exportados cargando el `.joblib` y enviando
**exactamente** las variables declaradas, en este orden:

| Modelo | Entrada (en orden) | Salida |
|--------|--------------------|--------|
| Dólar | `Dia` → `Inflacion` → `Tasa_interes` | `Precio_Dolar` (unidades monetarias) |
| Glucosa | `Edad` → `IMC` → `Actividad_Fisica` | `Nivel_Glucosa` (mg/dL) |
| Energía | `Temperatura` → `Hora` → `Dia_Semana` | `Consumo_Energia` (kWh) |

Rangos válidos que el frontend debe respetar al validar la entrada:

| Variable | Rango |
|----------|-------|
| `Dia` | 1 – 3650 |
| `Inflacion` | 0 – 1 (fracción decimal; 0.02 = 2 %) |
| `Tasa_interes` | 0 – 100 |
| `Edad` | 0 – 120 |
| `IMC` | 5 – 80 |
| `Actividad_Fisica` | 0 – 9 |
| `Temperatura` | −30 – 60 |
| `Hora` | 1 – 24 |
| `Dia_Semana` | 1 – 7 |

Las predicciones solo son fiables dentro del rango observado durante el
entrenamiento.

---

## Pruebas

```bash
pytest tests           # suite completa
pytest tests -v        # con detalle
```

| Archivo | Qué verifica |
|---------|--------------|
| `test_estructura.py` | Carpetas, módulos, contratos de datos, cabeceras reales de los CSV y **ausencia de cualquier componente de frontend** |
| `test_limpieza.py` | Reproducibilidad de la limpieza, ausencia de nulos/duplicados, rangos plausibles e **inmutabilidad de `data/raw`** |
| `test_modelado.py` | Partición reproducible y sin fuga de datos, coeficientes, inferencia contrastada contra un cálculo OLS independiente, cálculo correcto de β estandarizado |
| `test_evaluacion.py` | Métricas frente a cálculo manual, línea base, residuos, generación de gráficas y ausencia de mutación de datos |
| `test_modelos_exportados.py` | Carga de los `.joblib`, consistencia de predicciones, métricas guardadas y **contrato de integración** |

---

## Metodología aplicada

```text
data/raw/*.csv
    ↓  Comprensión de los datos   (auditoría de tipos, nulos, atípicos)
    ↓  Limpieza                   (reproducible en código, data/raw intacto)
data/processed/*_clean.csv
    ↓  Preparación                (tipado, rangos, separación X / y)
    ↓  Entrenamiento              (regresión lineal múltiple, 80/20, semilla 42)
    ↓  Evaluación                 (MSE/RMSE, R², residuos, línea base)
    ↓  Exportación                (models/*.joblib)
results/*/*.json, *.csv
```

### Decisiones metodológicas destacadas

1. **Los modelos `.joblib` existentes se verificaron antes de continuar.** Los
   tres se cargaron, se comprobó qué variables esperan y que predicen
   correctamente. Se confirmó que coinciden **exactamente** con el entrenamiento
   sobre la partición 80/20 (`random_state=42`), por lo que **se conservaron sin
   reentrenar**. La reestructuración no alteró ni un byte de los artefactos.

2. **Los valores atípicos no se eliminaron.** Se detectaron con el criterio IQR
   (× 1,5) y se documentaron, pero todos resultaron plausibles para su dominio,
   por lo que eliminarlos habría sesgado el modelo. Solo se descartan los valores
   *imposibles*.

3. **El impacto se compara con el coeficiente estandarizado (β), no con el
   coeficiente bruto.** Los coeficientes brutos no son comparables entre
   variables con escalas distintas. Este detalle es decisivo: usar los
   coeficientes brutos llevaría a conclusiones equivocadas.

4. **La codificación de `Hora` se decidió con evidencia, no por convención.**
   La convención sugeriría codificarla de forma cíclica (seno/coseno), pero los
   datos muestran que `Hora` es una rampa monótona (1→24) y no un ciclo horario.
   Codificarla con seno/coseno **reduce el R² de prueba de 0.8968 a 0.7851**, por
   lo que se conservó la escala lineal. La comparación queda registrada en
   `results/energia/limpieza.json`.

5. **La inferencia estadística se calcula sobre el conjunto de entrenamiento**,
   que es la muestra con la que se ajustó el modelo. Calcular errores estándar
   sobre otra muestra invalidaría los p-valores. Los resultados se verificaron
   contra una implementación de referencia independiente.

6. **Todos los datos originales son inmutables.** El pipeline lee siempre desde
   `data/raw/` y escribe en `data/processed/`; hay una prueba automatizada que lo
   verifica byte a byte.

---

## Reproducibilidad

Todos los resultados son reproducibles: la partición usa una semilla fija
(`random_state = 42`) y todos los artefactos de `results/` se generan a partir de
ella. Reejecutar los tres pipelines reproduce el proyecto de forma idéntica,
incluidos los modelos exportados.
