# Proyecto — Clasificación de Sentimiento (Aprendizaje de Máquina 2026-20)

## Parte 1: Machine Learning Clásico

Proyecto grupal de clasificación de reseñas de productos en español como
`negativo`, `neutral` o `positivo`, utilizando exclusivamente modelos clásicos
de scikit-learn y representaciones como Bag-of-Words, TF-IDF y n-gramas.

No se utilizan redes neuronales, transformers ni embeddings preentrenados,
de acuerdo con los requisitos de la Parte 1.

## Contenido del repositorio

```text
MLP1/
├── data/
│   ├── train.csv
│   ├── eval.csv
│   └── sample_submission.csv
├── notebooks/
│   ├── parte1_ml_clasico.ipynb
│   ├── parte1_ml_clasico.html
│   └── parte_1_word_char.ipynb
├── models/
│   ├── modelo_tfidf_logisticregression.joblib
│   └── modelo_alejandro_tfidf_word_char_lr_20261001_131902.joblib
├── submissions/
│   ├── submission_v1_tfidf_logisticregression.csv
│   └── submission_alejandro_tfidf_word_char_lr_20261001_131902.csv
└── build_notebook.py
```

Los nombres históricos de los archivos se conservan para identificar los
modelos y submissions correspondientes a los envíos realizados.

## Datos y evaluación

- `train.csv`: 12.000 reseñas con columnas `id`, `text` y `label`.
- `eval.csv`: 3.000 reseñas sin etiqueta, con columnas `id` y `text`.
- `sample_submission.csv`: formato de envío con columnas `id` y `answer`.
- Métrica principal: **accuracy**.

La comparación local utiliza una separación estratificada de `train.csv`:
85 % para entrenamiento y 15 % para validación, con `random_state=42`.
Esto corresponde a 10.200 ejemplos de entrenamiento y 1.800 de validación.

Los vectorizadores se ajustan dentro de los pipelines utilizando únicamente
los datos de entrenamiento. Después de seleccionar una configuración,
se reentrena con las 12.000 reseñas antes de predecir sobre `eval.csv`.

## Primera iteración: modelos clásicos

El notebook `parte1_ml_clasico.ipynb` incluye:

1. Carga y revisión de datos.
2. Análisis exploratorio.
3. Preprocesamiento de texto.
4. Comparación de representaciones y clasificadores.
5. Ajuste de hiperparámetros.
6. Evaluación y análisis de errores.
7. Entrenamiento final, generación del submission y guardado del modelo.

Se compararon ocho configuraciones, incluyendo Dummy, Naive Bayes,
regresión logística, LinearSVC, SGDClassifier y una variante ajustada
mediante `RandomizedSearchCV`.

La mejor configuración en validación fue:

- **Representación:** TF-IDF de unigramas y bigramas, `min_df=2`.
- **Clasificador:** `LogisticRegression`.
- **Accuracy local:** 0,7339.
- **Referencia Dummy:** 0,3511.

La variante ajustada mediante `RandomizedSearchCV` obtuvo 0,7250
en validación y no superó a la configuración inicial.

## Segunda iteración: palabras y caracteres

El notebook `parte_1_word_char.ipynb` complementa el trabajo inicial.
Reproduce el baseline y compara tres representaciones utilizando la misma
limpieza, separación de datos y configuración de regresión logística.

| Representación | Accuracy de validación |
|---|---:|
| TF-IDF de palabras | 0,7339 |
| TF-IDF de caracteres | 0,7328 |
| TF-IDF de palabras + caracteres | **0,7411** |

La configuración combinada utiliza:

- Palabras: `TfidfVectorizer(ngram_range=(1, 2), min_df=2)`.
- Caracteres: `TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)`.
- Combinación mediante `FeatureUnion`.
- Clasificador: `LogisticRegression(max_iter=2000, random_state=42)`.

La combinación consiguió 13 aciertos adicionales sobre los 1.800 ejemplos
de validación, reduciendo los errores de 479 a 466.

### Análisis de errores

En el baseline de palabras, 461 de los 479 errores correspondían a
confusiones entre positivo y negativo. La clase neutral tuvo 535 aciertos
de 536 ejemplos.

La revisión de seis errores mostró reseñas con opiniones contrapuestas,
cierres ambiguos y expresiones negativas que el modelo no clasificó
correctamente. Esta muestra permite formular hipótesis, pero no demuestra
que las etiquetas sean incorrectas.

Las comparaciones de este notebook son exploratorias y utilizan una única
partición de validación.

## Resultados públicos en Kaggle

Resultados registrados al **1 de octubre de 2026**:

| Envío | Score público |
|---|---:|
| v1 — TF-IDF + regresión logística | 0,73000 |
| v2 — N-gramas de caracteres | 0,73444 |
| v3 — Palabras + caracteres | 0,72666 |
| v4 — Características léxicas | 0,72555 |
| Nueva variante de palabras + caracteres | **0,73777** |

El último envío corresponde al archivo:

`submission_alejandro_tfidf_word_char_lr_20261001_131902.csv`

Sus predicciones difieren de las del envío v3 en 127 de las 3.000 reseñas.
Los archivos y configuraciones de v2, v3 y v4 están pendientes de
incorporarse al repositorio.

Se registraron cinco envíos completados. El mejor score público de los
envíos revisados supera el baseline del curso, de **0,65555**.

Los resultados locales y públicos corresponden a conjuntos diferentes.
El leaderboard público utiliza aproximadamente el 30 % de los datos de
evaluación; el resultado final depende del 70 % privado restante.

## Ejecución del notebook de palabras y caracteres

### Entorno utilizado

- Python 3.12.4.
- scikit-learn 1.4.2.
- pandas 2.2.2.

También se requieren NumPy, Matplotlib, Seaborn, joblib y un entorno
compatible con Jupyter. Queda pendiente registrar las versiones de todas
las dependencias.

### Pasos

1. Abrir `notebooks/parte_1_word_char.ipynb`.
2. Seleccionar el entorno de Python.
3. Ejecutar las celdas en orden.
4. Revisar las métricas de validación.
5. Ejecutar las celdas finales para entrenar con todo `train.csv` y guardar
   el modelo y la submission.

El notebook localiza los datos desde la raíz del repositorio o desde la
carpeta `notebooks`. Los archivos generados incorporan una marca de tiempo
para distinguir ejecuciones.

### Guardado y recarga

El modelo combinado se guarda con `joblib` e incluye la limpieza de texto.

Para recargarlo en otra sesión deben estar definidas las funciones
`clean_text` y `clean_text_batch`, incluidas en el notebook.

Se comprobó que el modelo recargado reproduce las predicciones del CSV
en la misma sesión. Queda pendiente verificar la recarga en una sesión nueva.

Antes de guardar la submission se comprueban:

- Columnas `id` y `answer`.
- Cantidad de filas.
- Identificadores únicos y coincidentes con la plantilla.
- Etiquetas válidas.
- Ausencia de valores nulos.

## Próximos pasos

1. Incorporar los archivos y configuraciones de los envíos anteriores
   que faltan en el repositorio.
2. Verificar la recarga del modelo en una sesión nueva y completar
   el registro de dependencias.
3. Desarrollar en otro notebook el experimento de representación adicional
   de la última oración, documentando su comparación mediante validación cruzada.
4. Mantener una bitácora grupal de experimentos, resultados y contribuciones.
5. Preparar la sustentación y la entrega del notebook y modelo
   correspondientes al mejor envío público del grupo.

## Notas

- Estos experimentos corresponden a la **Parte 1**, aunque estén distribuidos
  en varios notebooks.
- La rúbrica exige al menos cinco envíos diferentes que evidencien intentos
  de mejora; no exige que cada envío supere al anterior.
- `build_notebook.py` genera el notebook inicial. Ejecutarlo puede sobrescribir
  cambios manuales y las salidas guardadas de ese notebook.