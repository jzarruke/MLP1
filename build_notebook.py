"""
Genera el notebook de la Parte 1 (ML clásico) de la competencia de
clasificación de sentimiento (Aprendizaje de Máquina 2026-20).
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

# ---------------------------------------------------------------------------
# 0. Portada / contexto
# ---------------------------------------------------------------------------
md("""\
# Proyecto — Clasificación de Sentimiento en Reseñas de Producto
## Parte 1: Machine Learning Clásico

**Curso:** Aprendizaje de Máquina 2026-20 — Universidad de los Andes
**Competencia:** Kaggle — clasificación de reseñas en `negativo`, `neutral`, `positivo`

### Objetivo de esta entrega

Construir un clasificador de sentimiento usando **únicamente** técnicas clásicas de
aprendizaje automático (scikit-learn) y representaciones de texto tipo *bag-of-words* /
TF-IDF / n-gramas, sin redes neuronales ni embeddings preentrenados (requisito de la
Parte 1 del enunciado).

### Contenido del notebook

1. Carga de datos
2. Análisis exploratorio (EDA)
3. Preprocesamiento de texto
4. Vectorización (Bag-of-Words vs. TF-IDF)
5. Modelos base y comparación
6. Ajuste de hiperparámetros (GridSearchCV)
7. Evaluación y análisis de errores del mejor modelo
8. Entrenamiento final + generación del `submission.csv` para Kaggle
9. Guardado del modelo (`joblib`)
10. Ideas para seguir subiendo el score en próximas iteraciones

> **Nota:** este notebook está pensado como punto de partida sólido y bien documentado
> para la competencia. La idea es usarlo como base y seguir iterando (probar más
> combinaciones de n-gramas, features adicionales, balanceo de clases, etc.) para ir
> subiendo de percentil en el leaderboard, dado que la rúbrica exige al menos 5 envíos
> distintos que muestren una mejora progresiva.
""")

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
md("## 1. Imports y configuración")

code("""\
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
RANDOM_STATE = 42

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
SUBMISSIONS_DIR = Path("submissions")
for d in (MODELS_DIR, SUBMISSIONS_DIR):
    d.mkdir(exist_ok=True)
""")

# ---------------------------------------------------------------------------
# 2. Carga de datos
# ---------------------------------------------------------------------------
md("""\
## 2. Carga de datos

- `train.csv`: 12,000 reseñas etiquetadas (`id`, `text`, `label`).
- `eval.csv`: 3,000 reseñas sin etiquetar, es el conjunto que se sube a Kaggle.
- `sample_submission.csv`: formato exacto que espera Kaggle (`id`, `answer`).
""")

code("""\
train_df = pd.read_csv(DATA_DIR / "train.csv")
eval_df = pd.read_csv(DATA_DIR / "eval.csv")
sample_submission = pd.read_csv(DATA_DIR / "sample_submission.csv")

print("train:", train_df.shape)
print("eval:", eval_df.shape)
print("sample_submission:", sample_submission.shape)
train_df.head()
""")

code("""\
# Consistencia de las etiquetas esperadas por Kaggle (ver PDF de instrucciones, sección 3.3)
LABELS = ["negativo", "neutral", "positivo"]
assert set(train_df["label"].unique()) == set(LABELS), train_df["label"].unique()
print("Valores nulos en train:\\n", train_df.isna().sum())
print("Valores nulos en eval:\\n", eval_df.isna().sum())
print("Reseñas duplicadas en train (por texto):", train_df["text"].duplicated().sum())
print("Reseñas duplicadas en eval (por texto):", eval_df["text"].duplicated().sum())
""")

# ---------------------------------------------------------------------------
# 3. EDA
# ---------------------------------------------------------------------------
md("""\
## 3. Análisis exploratorio de datos (EDA)

### 3.1 Balance de clases

Es importante revisar el balance de clases porque, si está desbalanceado, la exactitud
(accuracy, la métrica de la competencia) puede ser engañosa y conviene usar
`class_weight="balanced"` o estratificar cuidadosamente los splits.
""")

code("""\
fig, ax = plt.subplots(figsize=(6, 4))
order = train_df["label"].value_counts().index
sns.countplot(data=train_df, x="label", order=order, hue="label",
              palette="viridis", legend=False, ax=ax)
for container in ax.containers:
    ax.bar_label(container)
ax.set_title("Distribución de clases en train.csv")
ax.set_xlabel("Sentimiento")
ax.set_ylabel("Número de reseñas")
plt.tight_layout()
plt.show()

train_df["label"].value_counts(normalize=True).round(3)
""")

md("""\
Las clases están **levemente desbalanceadas** (≈35% negativo, ≈35% positivo, ≈30%
neutral), no es un desbalance severo, pero lo tendremos en cuenta al comparar modelos
(usaremos `class_weight="balanced"` como una de las variantes a probar) y al hacer el
split usaremos `stratify` para mantener esta proporción en train/validación.
""")

md("### 3.2 Longitud de las reseñas (caracteres y palabras)")

code("""\
train_df["n_chars"] = train_df["text"].str.len()
train_df["n_words"] = train_df["text"].str.split().str.len()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(data=train_df, x="n_words", hue="label", element="step",
             stat="density", common_norm=False, ax=axes[0])
axes[0].set_title("Distribución de longitud (palabras) por clase")

sns.boxplot(data=train_df, x="label", y="n_chars", hue="label",
            legend=False, ax=axes[1])
axes[1].set_title("Longitud en caracteres por clase")
plt.tight_layout()
plt.show()

train_df.groupby("label")[["n_chars", "n_words"]].describe().T
""")

md("""\
La longitud de las reseñas es muy similar entre clases (mediana ~52-55 palabras), por lo
que la longitud del texto **no es, por sí sola, una señal útil** para distinguir el
sentimiento — el modelo tiene que aprender del contenido léxico y no solo de cuánto
escribe el usuario.
""")

md("### 3.3 Palabras y n-gramas más frecuentes por clase")

code("""\
SPANISH_STOPWORDS = set(\"\"\"
de la que el en y a los del se las por un para con no una su al lo como más pero sus le
ya o este sí porque esta entre cuando muy sin sobre también me hasta hay donde quien
desde todo nos durante todos uno les ni contra otros ese eso ante ellos e esto mi antes
algunos qué unos yo otro otras otra él tanto esa estos mucho quienes nada muchos cual
poco ella estar estas algunas algo nosotros mi mis tú te ti tu tus ellas nosotras
vosotros vosotras os mío mía míos mías tuyo tuya tuyos tuyas suyo suya suyos suyas
nuestro nuestra nuestros nuestras vuestro vuestra vuestros vuestras esos esas fue ser
es son era eran son fui fuiste lo un una unos unas
\"\"\".split())

def top_ngrams(texts, n=20, ngram_range=(1, 1), stopwords=SPANISH_STOPWORDS):
    vec = CountVectorizer(ngram_range=ngram_range, stop_words=list(stopwords))
    X = vec.fit_transform(texts.str.lower())
    freqs = np.asarray(X.sum(axis=0)).ravel()
    vocab = vec.get_feature_names_out()
    top_idx = np.argsort(freqs)[::-1][:n]
    return pd.Series(freqs[top_idx], index=vocab[top_idx])

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
for ax, label in zip(axes, LABELS):
    subset = train_df.loc[train_df["label"] == label, "text"]
    top = top_ngrams(subset, n=15, ngram_range=(1, 1))
    sns.barplot(x=top.values, y=top.index, hue=top.index, legend=False,
                palette="mako", ax=ax)
    ax.set_title(f"Top unigramas — {label}")
plt.tight_layout()
plt.show()
""")

code("""\
# Bigramas más frecuentes por clase: suelen capturar mejor negaciones y matices
# ("no funciona", "muy buena", "no vale") que los unigramas sueltos.
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, label in zip(axes, LABELS):
    subset = train_df.loc[train_df["label"] == label, "text"]
    top = top_ngrams(subset, n=15, ngram_range=(2, 2))
    sns.barplot(x=top.values, y=top.index, hue=top.index, legend=False,
                palette="flare", ax=ax)
    ax.set_title(f"Top bigramas — {label}")
plt.tight_layout()
plt.show()
""")

md("### 3.4 Reseñas con sentimiento mixto (elogios + quejas)")

code("""\
# El enunciado advierte explícitamente que muchas reseñas mezclan elogios y quejas,
# y que el sentimiento global depende del orden, la negación y los conectores
# contrastivos ("pero", "sin embargo", "aunque"). Veamos ejemplos reales.
contrast_markers = r"\\b(pero|sin embargo|aunque|no obstante|eso sí)\\b"
mixed = train_df[train_df["text"].str.contains(contrast_markers, case=False, regex=True)]
print(f"{len(mixed)} de {len(train_df)} reseñas ({len(mixed)/len(train_df):.1%}) "
      f"contienen un conector contrastivo.")
mixed["label"].value_counts(normalize=True).round(3)
""")

code("""\
for label in LABELS:
    ejemplo = mixed[mixed["label"] == label]["text"].iloc[0]
    print(f"--- {label.upper()} ---")
    print(ejemplo[:300])
    print()
""")

md("""\
Casi un tercio de las reseñas usa conectores contrastivos, y aparecen en las tres
clases — confirma lo que dice el enunciado: **no basta con contar palabras positivas y
negativas**, el modelo necesita algo de contexto local. Con bag-of-words puro esto es
difícil de capturar del todo, pero los **n-gramas (bigramas/trigramas)** ayudan a que
`"no me gustó"` o `"pero funciona"` se traten como unidades, en vez de perder la negación
al separar las palabras.
""")

# ---------------------------------------------------------------------------
# 4. Preprocesamiento
# ---------------------------------------------------------------------------
md("""\
## 4. Preprocesamiento de texto

Usamos una limpieza **ligera**: a propósito no eliminamos tildes ni negaciones, porque
son señal relevante para el sentimiento (`"esta"` vs. `"está"`, `"no"`, `"nunca"`,
`"sin"`). Tampoco quitamos stopwords dentro del pipeline final del vectorizador (se
prueba como hiperparámetro), porque partículas como `"no"` o `"muy"` suelen ser
stopwords genéricas pero aquí son muy informativas.
""")

code("""\
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\\S+|www\\.\\S+", " ", text)          # URLs
    text = re.sub(r"[^a-zA-ZÀ-ÿñÑ0-9¡!¿?.,;: ]", " ", text)  # ruido / emojis / símbolos raros
    text = re.sub(r"\\s+", " ", text).strip()
    return text

train_df["text_clean"] = train_df["text"].apply(clean_text)
eval_df["text_clean"] = eval_df["text"].apply(clean_text)

train_df[["text", "text_clean"]].head(3)
""")

md("""\
### Split de entrenamiento / validación

Como `eval.csv` no tiene etiquetas (es el conjunto de Kaggle), separamos una porción de
`train.csv` como validación local, **estratificada** por clase, para poder comparar
modelos de forma justa antes de generar cualquier envío.
""")

code("""\
X_train_text, X_val_text, y_train, y_val = train_test_split(
    train_df["text_clean"],
    train_df["label"],
    test_size=0.15,
    stratify=train_df["label"],
    random_state=RANDOM_STATE,
)
print("Entrenamiento:", X_train_text.shape[0], "  Validación:", X_val_text.shape[0])
y_train.value_counts(normalize=True).round(3)
""")

# ---------------------------------------------------------------------------
# 5. Vectorización + modelos base
# ---------------------------------------------------------------------------
md("""\
## 5. Vectorización y modelos base

Comparamos **Bag-of-Words (conteos)** vs. **TF-IDF**, y varios clasificadores clásicos
permitidos por el enunciado: `MultinomialNB`, `LogisticRegression`, `LinearSVC` y
`SGDClassifier` (variante lineal, útil como referencia rápida). Incluimos también un
`DummyClassifier` como piso de referencia: cualquier modelo real debe superarlo con
claridad.
""")

code("""\
# Guardamos también el pipeline ya ajustado de cada candidato, para poder
# comparar objetivamente al final (sección 8) cuál fue realmente el mejor,
# en lugar de asumir a ciegas que el resultado del tuning es superior.
fitted_models = {}

def evaluate(pipeline, name, X_tr=X_train_text, y_tr=y_train, X_va=X_val_text, y_va=y_val):
    pipeline.fit(X_tr, y_tr)
    preds = pipeline.predict(X_va)
    acc = accuracy_score(y_va, preds)
    fitted_models[name] = pipeline
    return {"modelo": name, "accuracy_val": acc}

results = []

# Piso de referencia
dummy = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", DummyClassifier(strategy="most_frequent")),
])
results.append(evaluate(dummy, "Dummy (clase mayoritaria)"))

candidatos = {
    "BoW + MultinomialNB": Pipeline([
        ("vec", CountVectorizer(ngram_range=(1, 2), min_df=2)),
        ("clf", MultinomialNB()),
    ]),
    "TFIDF + MultinomialNB": Pipeline([
        ("vec", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("clf", MultinomialNB()),
    ]),
    "TFIDF + LogisticRegression": Pipeline([
        ("vec", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ]),
    "TFIDF + LogisticRegression (balanced)": Pipeline([
        ("vec", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                    random_state=RANDOM_STATE)),
    ]),
    "TFIDF + LinearSVC": Pipeline([
        ("vec", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("clf", LinearSVC(random_state=RANDOM_STATE)),
    ]),
    "TFIDF + SGDClassifier": Pipeline([
        ("vec", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("clf", SGDClassifier(loss="hinge", random_state=RANDOM_STATE)),
    ]),
}

for name, pipe in candidatos.items():
    results.append(evaluate(pipe, name))

results_df = pd.DataFrame(results).sort_values("accuracy_val", ascending=False)
results_df
""")

code("""\
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=results_df, x="accuracy_val", y="modelo", hue="modelo",
            palette="crest", legend=False, ax=ax)
ax.set_xlim(0, 1)
ax.set_title("Accuracy en validación por modelo (configuración por defecto)")
for i, v in enumerate(results_df["accuracy_val"]):
    ax.text(v + 0.01, i, f"{v:.3f}", va="center")
plt.tight_layout()
plt.show()
""")

# ---------------------------------------------------------------------------
# 6. Tuning
# ---------------------------------------------------------------------------
md("""\
## 6. Ajuste de hiperparámetros (GridSearchCV)

Tomamos la(s) mejor(es) combinación(es) de la comparación anterior y afinamos
hiperparámetros clave del vectorizador (`ngram_range`, `min_df`, `max_features`, uso de
stopwords) y del clasificador (`C`), usando validación cruzada estratificada de 5 folds
sobre el conjunto de entrenamiento (no sobre el de validación, para no filtrar
información).
""")

code("""\
pipe = Pipeline([
    ("vec", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
])

param_grid = {
    "vec__ngram_range": [(1, 1), (1, 2), (1, 3)],
    "vec__min_df": [1, 2, 3],
    "vec__max_features": [20000, 40000, None],
    "vec__sublinear_tf": [True, False],
    "clf__C": [0.1, 1, 3, 10],
    "clf__class_weight": [None, "balanced"],
}

# Búsqueda amplia -> RandomizedSearch para no disparar el tiempo de cómputo,
# seguida de un ajuste fino con GridSearch alrededor del mejor punto encontrado.
from sklearn.model_selection import RandomizedSearchCV

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
search = RandomizedSearchCV(
    pipe, param_distributions=param_grid, n_iter=25, cv=cv,
    scoring="accuracy", random_state=RANDOM_STATE, n_jobs=-1, verbose=1,
)
search.fit(X_train_text, y_train)

print("Mejor accuracy (CV, train):", search.best_score_)
print("Mejores hiperparámetros:", search.best_params_)
""")

code("""\
tuned_pipe = search.best_estimator_
tuned_val_preds = tuned_pipe.predict(X_val_text)
tuned_val_acc = accuracy_score(y_val, tuned_val_preds)
print(f"Accuracy en validación con el pipeline tuneado: {tuned_val_acc:.4f}")

fitted_models["TFIDF + LogisticRegression (tuned)"] = tuned_pipe
results_df = pd.concat([
    results_df,
    pd.DataFrame([{"modelo": "TFIDF + LogisticRegression (tuned)", "accuracy_val": tuned_val_acc}]),
], ignore_index=True).sort_values("accuracy_val", ascending=False).reset_index(drop=True)
results_df
""")

md("""\
**Importante:** el tuning optimiza accuracy promedio en *cross-validation sobre el
conjunto de entrenamiento*, lo cual no garantiza que ese configuración sea también la
mejor sobre nuestro split de validación (puede haber algo de varianza, sobre todo con
`RandomizedSearchCV` explorando pocas combinaciones). Por eso, en vez de asumir que el
modelo tuneado es automáticamente el mejor, **comparamos explícitamente contra todos los
candidatos evaluados** y elegimos el de mayor accuracy en validación para continuar.
""")

code("""\
best_model_name = results_df.iloc[0]["modelo"]
best_val_acc = results_df.iloc[0]["accuracy_val"]
best_pipe = fitted_models[best_model_name]
print(f"Modelo seleccionado: {best_model_name}  (accuracy validación = {best_val_acc:.4f})")
""")

code("""\
val_preds = best_pipe.predict(X_val_text)
assert accuracy_score(y_val, val_preds) == best_val_acc
""")

# ---------------------------------------------------------------------------
# 7. Evaluación y análisis de errores
# ---------------------------------------------------------------------------
md("## 7. Evaluación del mejor modelo y análisis de errores")

code("""\
print(classification_report(y_val, val_preds, target_names=LABELS))
""")

code("""\
fig, ax = plt.subplots(figsize=(5.5, 5))
ConfusionMatrixDisplay.from_predictions(
    y_val, val_preds, labels=LABELS, cmap="Blues", ax=ax, colorbar=False,
)
ax.set_title("Matriz de confusión — validación")
plt.tight_layout()
plt.show()
""")

md("""\
La mayoría de la confusión ocurre entre **neutral** y las clases polarizadas, lo cual
es esperable: `neutral` es la clase "intermedia" y muchas veces depende de matices finos
(intensidad, presencia de una sola queja menor) que un modelo lineal sobre TF-IDF no
siempre captura. Veamos ejemplos concretos mal clasificados, en particular los que tienen
conectores contrastivos.
""")

code("""\
val_df = pd.DataFrame({
    "text": train_df.loc[X_val_text.index, "text"].values,
    "real": y_val.values,
    "pred": val_preds,
})
errores = val_df[val_df["real"] != val_df["pred"]]
print(f"{len(errores)} errores de {len(val_df)} ejemplos de validación "
      f"({len(errores)/len(val_df):.1%})")

errores_contraste = errores[errores["text"].str.contains(contrast_markers, case=False, regex=True)]
print(f"De esos, {len(errores_contraste)} contienen un conector contrastivo.")
errores_contraste.sample(min(5, len(errores_contraste)), random_state=RANDOM_STATE)
""")

md("""\
Esto confirma la limitación que anticipa el enunciado: los modelos de bolsa de
palabras / TF-IDF no capturan bien el **orden** ni el **alcance** de la negación o los
contrastes (`"la batería es excelente, pero la cámara resultó decepcionante"`). Es
exactamente el tipo de caso donde se espera que los modelos secuenciales de la Parte 2
(RNN/LSTM/CNN) mejoren sobre esta línea base.
""")

# ---------------------------------------------------------------------------
# 8. Entrenamiento final + submission
# ---------------------------------------------------------------------------
md("""\
## 8. Entrenamiento final sobre todos los datos y predicción sobre `eval.csv`

Una vez elegida la configuración, reentrenamos con **todo** `train.csv` (train +
validación) para aprovechar al máximo los datos antes de predecir sobre el conjunto de
evaluación de Kaggle.
""")

code("""\
from sklearn.base import clone

# Re-partimos de una copia SIN entrenar de la mejor configuración encontrada
# (best_pipe puede venir de la comparación inicial o del tuning, según cuál haya
# ganado en la sección anterior) para entrenarla desde cero con todos los datos.
final_pipe = clone(best_pipe)
final_pipe.fit(train_df["text_clean"], train_df["label"])

eval_preds = final_pipe.predict(eval_df["text_clean"])
pd.Series(eval_preds).value_counts(normalize=True).round(3)
""")

code("""\
submission = pd.DataFrame({"id": eval_df["id"], "answer": eval_preds})

# Verificación de formato contra sample_submission.csv antes de guardar
assert list(submission.columns) == list(sample_submission.columns)
assert len(submission) == len(sample_submission)
assert set(submission["answer"].unique()) <= set(LABELS)
assert (submission["id"].values == sample_submission["id"].values).all()

model_slug = re.sub(r"[^a-z0-9]+", "_", best_model_name.lower()).strip("_")
submission_path = SUBMISSIONS_DIR / f"submission_v1_{model_slug}.csv"
submission.to_csv(submission_path, index=False)
print("Guardado:", submission_path)
submission.head()
""")

# ---------------------------------------------------------------------------
# 9. Guardar modelo
# ---------------------------------------------------------------------------
md("""\
## 9. Guardado del modelo entrenado

Se guarda con `joblib`, como pide el enunciado (sección 4.2), junto con los
hiperparámetros elegidos para que el entrenamiento sea reproducible.
""")

code("""\
model_path = MODELS_DIR / f"modelo_{model_slug}.joblib"
joblib.dump(final_pipe, model_path)
print("Modelo guardado en:", model_path)
print("Hiperparámetros del mejor pipeline:")
final_pipe.get_params()["vec"], final_pipe.get_params()["clf"]
""")

code("""\
# Verificación de que el modelo guardado se puede recargar y predice igual
# (comparamos contra las predicciones del propio final_pipe sobre eval, ya que
# fue entrenado con train+validación combinados y por eso no es comparable con val_preds)
reloaded = joblib.load(model_path)
assert (reloaded.predict(eval_df["text_clean"]) == eval_preds).all()
print("OK: el modelo recargado reproduce las mismas predicciones sobre eval.csv.")
""")

# ---------------------------------------------------------------------------
# 10. Próximos pasos
# ---------------------------------------------------------------------------
md("## 10. Resumen y próximos pasos para seguir subiendo el score")

code("""\
from IPython.display import Markdown, display

resumen = results_df.reset_index(drop=True).copy()
resumen["accuracy_val"] = resumen["accuracy_val"].round(4)
display(Markdown("**Resultado de esta primera iteración (accuracy en validación):**"))
display(resumen)
""")

md("""\
La rúbrica exige al menos **5 envíos distintos** a Kaggle mostrando que el grupo intentó
mejorar su enfoque inicial. Ideas concretas para las próximas iteraciones (manteniéndonos
dentro de las reglas de la Parte 1: nada de redes neuronales ni embeddings
preentrenados):

1. **Character n-grams** (`analyzer="char_wb"`, ngramas 3-5) en TF-IDF: suelen ayudar
   mucho con errores ortográficos y variaciones informales típicas de reseñas reales.
2. **Combinar word n-grams + char n-grams** con `FeatureUnion`.
3. **Features léxicas hechas a mano**: número de signos de exclamación/interrogación,
   presencia de conectores contrastivos (`pero`, `sin embargo`), conteo de palabras en
   mayúsculas, longitud — concatenadas como columnas extra a las features de TF-IDF.
4. **Probar SVM con kernel** (`SVC` con `kernel="linear"` calibrado, o `CalibratedClassifierCV`
   sobre `LinearSVC` para tener probabilidades) y comparar contra Logistic Regression.
5. **Manejo del desbalance**: sobre-muestreo de la clase `neutral` (p. ej. con
   `RandomOverSampler` de `imbalanced-learn`) ya que es la que más se confunde.
6. **Ensamble por votación** (`VotingClassifier`) combinando Logistic Regression, SVM y
   Naive Bayes, que suelen equivocarse en casos distintos.
7. Ampliar la búsqueda de hiperparámetros (`GridSearchCV` completo alrededor del mejor
   punto encontrado por `RandomizedSearchCV`).

> **Aporte individual / trabajo en grupo:** este notebook es la base común del equipo;
> cada envío adicional a Kaggle debería documentarse aquí (o en una sección de bitácora)
> indicando quién lo hizo y qué cambió, para sustentar el aporte individual de cada
> integrante en la evaluación.

---

## Bitácora de envíos a Kaggle

| # | Fecha | Modelo | Accuracy validación local | Score público Kaggle |
|---|---|---|---|---|
| 1 | 2026-09-12 | TF-IDF (uni+bigramas) + LogisticRegression | 0.7339 | **0.73000** (baseline: 0.65555) |
""")

# ---------------------------------------------------------------------------
# 11. Iteración 2: variantes para subir el score
# ---------------------------------------------------------------------------
md("""\
## 11. Iteración 2 — probando variantes para subir el score

El envío 1 (TF-IDF palabras + Logistic Regression) dio **0.73000** en el leaderboard
público, bien por encima del baseline (0.65555). Ahora probamos, sobre el mismo split
de validación para que la comparación sea justa, las ideas que dejamos anotadas en la
sección anterior. Seguimos dentro de las reglas de la Parte 1 (nada de redes
neuronales ni embeddings preentrenados).
""")

md("""\
### 11.1 TF-IDF de n-gramas de caracteres

Los n-gramas de caracteres (`analyzer="char_wb"`) son robustos a errores de ortografía,
alargamientos ("buenísimoo"), y variaciones informales, porque no dependen de que la
palabra completa coincida exactamente con el vocabulario visto en entrenamiento.
""")

code("""\
char_pipe = Pipeline([
    ("vec", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                             max_features=50000, sublinear_tf=True)),
    ("clf", LogisticRegression(max_iter=3000, C=3, random_state=RANDOM_STATE)),
])
results.append(evaluate(char_pipe, "TFIDF char(3-5) + LogisticRegression"))
results[-1]
""")

md("### 11.2 Combinación de n-gramas de palabras y de caracteres (`FeatureUnion`)")

code("""\
word_char_pipe = Pipeline([
    ("features", FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                  max_features=30000)),
    ])),
    ("clf", LogisticRegression(max_iter=3000, C=3, random_state=RANDOM_STATE)),
])
results.append(evaluate(word_char_pipe, "TFIDF word+char + LogisticRegression"))
results[-1]
""")

md("""\
### 11.3 Features léxicas hechas a mano + TF-IDF

Agregamos señales explícitas que la intuición y el EDA sugieren que importan: signos de
exclamación/interrogación, presencia de conectores contrastivos, longitud y longitud
promedio de palabra. Se calculan sobre `text_clean` (ya limpio, pero conserva
`¡ ! ¿ ? . , ; :`).
""")

code("""\
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix

class LexicalFeatures(BaseEstimator, TransformerMixin):
    \"\"\"Features numéricas simples calculadas sobre el texto limpio.\"\"\"

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        s = pd.Series(X).reset_index(drop=True)
        n_exclaim = s.str.count("!")
        n_question = s.str.count(r"\\?")
        has_contrast = s.str.contains(contrast_markers, regex=True).astype(int)
        n_words = s.str.split().str.len().fillna(0)

        def avg_word_len(text):
            words = re.findall(r"[a-zA-ZÀ-ÿñÑ]+", text)
            return np.mean([len(w) for w in words]) if words else 0.0

        avg_len = s.apply(avg_word_len)
        feats = np.column_stack([n_exclaim, n_question, has_contrast, n_words, avg_len])
        return csr_matrix(feats)

lexical_pipe = Pipeline([
    ("features", FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)),
        ("lexical", LexicalFeatures()),
    ])),
    ("clf", LogisticRegression(max_iter=3000, C=3, random_state=RANDOM_STATE)),
])
results.append(evaluate(lexical_pipe, "TFIDF word + features léxicas + LogisticRegression"))
results[-1]
""")

md("""\
### 11.4 Sobre-muestreo (oversampling) de la clase `neutral`

`neutral` es la clase minoritaria y la que más se confunde (ver matriz de confusión de
la sección 7). Probamos `RandomOverSampler` **solo sobre el conjunto de
entrenamiento** (nunca sobre validación, para no inflar artificialmente la métrica) con
un `Pipeline` de `imbalanced-learn`, que se asegura de que el resampling ocurra
únicamente durante `fit`.
""")

code("""\
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

oversample_pipe = ImbPipeline([
    ("vec", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)),
    ("oversample", RandomOverSampler(random_state=RANDOM_STATE)),
    ("clf", LogisticRegression(max_iter=3000, C=3, random_state=RANDOM_STATE)),
])
results.append(evaluate(oversample_pipe, "TFIDF + oversampling (neutral) + LogisticRegression"))
results[-1]
""")

md("""\
### 11.5 Ensamble por votación

Combinamos tres modelos que suelen equivocarse en casos distintos: Logistic
Regression, Naive Bayes y SVM lineal. Usamos `voting="hard"` (voto mayoritario) para no
depender de calibrar probabilidades sobre `LinearSVC`.
""")

code("""\
from sklearn.ensemble import VotingClassifier

ensemble_pipe = Pipeline([
    ("vec", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)),
    ("clf", VotingClassifier(estimators=[
        ("lr", LogisticRegression(max_iter=3000, C=3, random_state=RANDOM_STATE)),
        ("nb", MultinomialNB()),
        ("svm", LinearSVC(random_state=RANDOM_STATE)),
    ], voting="hard")),
])
results.append(evaluate(ensemble_pipe, "Ensamble (LR + NB + SVM, voto mayoritario)"))
results[-1]
""")

md("### 11.6 Comparación de todos los candidatos (iteración 1 + iteración 2)")

code("""\
results_df = pd.DataFrame(results).drop_duplicates(subset="modelo", keep="last") \\
    .sort_values("accuracy_val", ascending=False).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(9, 6))
sns.barplot(data=results_df, x="accuracy_val", y="modelo", hue="modelo",
            palette="crest", legend=False, ax=ax)
ax.set_xlim(0, 1)
ax.set_title("Accuracy en validación — todos los candidatos probados")
for i, v in enumerate(results_df["accuracy_val"]):
    ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
plt.tight_layout()
plt.show()

results_df
""")

code("""\
best_model_name = results_df.iloc[0]["modelo"]
best_val_acc = results_df.iloc[0]["accuracy_val"]
best_pipe = fitted_models[best_model_name]
print(f"Mejor modelo de la iteración 2: {best_model_name}  "
      f"(accuracy validación = {best_val_acc:.4f}, vs. 0.7339 del envío 1)")
""")

md("""\
### 11.7 Generar varios envíos distintos para Kaggle

La rúbrica pide al menos 5 envíos distintos mostrando que el grupo iteró. Ya usamos el
envío 1. Acá generamos un envío por cada variante nueva de esta iteración (no solo la
ganadora), entrenada sobre **todo** `train.csv`, para tener varias entregas legítimas y
distintas entre sí — así, aunque alguna no mejore el score público, se documenta el
intento y el porqué.
""")

code("""\
from sklearn.base import clone

nuevas_variantes = {
    "v2_char_ngrams": "TFIDF char(3-5) + LogisticRegression",
    "v3_word_char": "TFIDF word+char + LogisticRegression",
    "v4_lexical": "TFIDF word + features léxicas + LogisticRegression",
    "v5_oversample_neutral": "TFIDF + oversampling (neutral) + LogisticRegression",
    "v6_ensemble": "Ensamble (LR + NB + SVM, voto mayoritario)",
}

resumen_envios = []
for slug, model_name in nuevas_variantes.items():
    pipe = clone(fitted_models[model_name])
    pipe.fit(train_df["text_clean"], train_df["label"])
    preds = pipe.predict(eval_df["text_clean"])

    sub = pd.DataFrame({"id": eval_df["id"], "answer": preds})
    assert list(sub.columns) == list(sample_submission.columns)
    assert len(sub) == len(sample_submission)
    assert (sub["id"].values == sample_submission["id"].values).all()
    assert set(sub["answer"].unique()) <= set(LABELS)

    path = SUBMISSIONS_DIR / f"submission_{slug}.csv"
    sub.to_csv(path, index=False)

    model_path_i = MODELS_DIR / f"modelo_{slug}.joblib"
    joblib.dump(pipe, model_path_i)

    val_acc_i = results_df.set_index("modelo").loc[model_name, "accuracy_val"]
    resumen_envios.append({
        "archivo": path.name, "modelo": model_name, "accuracy_val": round(val_acc_i, 4),
    })
    print(f"OK -> {path.name}  (val_acc={val_acc_i:.4f})")

pd.DataFrame(resumen_envios)
""")

md("""\
**Cómo usar estos envíos:** súbelos a Kaggle uno por uno (no todos a la vez, para poder
ver cómo se mueve el score público con cada cambio) y anota el resultado real en la
tabla de la bitácora de esta sección. Con esto, entre el envío 1 y estos 5 nuevos ya
quedan cubiertos los 5 envíos distintos que exige la rúbrica — y de paso queda evidencia
clara de qué técnicas sí ayudaron y cuáles no, que es justo lo que pide "calidad del
proceso" en la rúbrica.

Si alguno de estos supera el 0.73000 del envío 1, ese pasa a ser el modelo "oficial" del
grupo para la entrega de la Parte 1 (el que se debe subir a Bloque Neón en la semana 11,
junto con el notebook).

### Actualización — envíos reales en Kaggle (iteración 2)

| # | Modelo | Accuracy validación local | Score público Kaggle |
|---|---|---|---|
| 1 | TF-IDF (uni+bigramas) + LogisticRegression | 0.7339 | 0.73000 |
| 2 | TF-IDF char(3-5) + LogisticRegression | 0.7400 | 0.73444 |
| 3 | TF-IDF word+char + LogisticRegression | 0.7400 | 0.72666 |
| 4 | TF-IDF word + features léxicas + LogisticRegression | 0.7256 | 0.72555 |

Hallazgo importante: **v2 y v3 empataron en validación local (0.7400) pero en Kaggle v3
quedó peor que v1.** La combinación word+char, con muchas más features, se ajustó un
poco más a nuestro split local sin generalizar igual de bien al leaderboard real — buena
evidencia de que "más features" no es automáticamente mejor, y de por qué conviene
confirmar siempre con un envío real antes de fijar el modelo oficial.
""")

# ---------------------------------------------------------------------------
# 12. Iteración 3: el poder de la última oración
# ---------------------------------------------------------------------------
md("""\
## 12. Iteración 3 — explotando la estructura de las reseñas

Con un equipo del curso llegando a 0.88 en el leaderboard, el margen de mejora sugiere
que falta algo más estructural que afinar hiperparámetros. Revisando varios ejemplos a
mano (ver EDA, sección 3.4) se nota un patrón: **las reseñas siguen una plantilla**
—primero un párrafo neutral sobre el pedido/envío/empaque, y al final una o dos
oraciones con la opinión real— y esto aplica en las tres clases:

> *"Lo tengo en el escritorio del trabajo y por ahora nomás lo uso de vez en cuando.
> Viene con su cable y el manual en la caja."* (neutral)
>
> *"El diseño se sintió durable desde el primer día. Aun así, la relación
> calidad-precio se ve desagradable con el tiempo."* (negativo — el "aun así" marca el
> giro hacia la opinión real)

Si el sentimiento está concentrado en la(s) última(s) oración(es), entonces todo el
párrafo de logística que viene antes es **ruido** para el vectorizador de TF-IDF: diluye
las frecuencias de las palabras que sí importan. La hipótesis a probar: un modelo
entrenado *solo con la última oración* debería superar a uno entrenado con la reseña
completa.
""")

code("""\
def last_chunk(text: str, n_sent: int = 1) -> str:
    \"\"\"Devuelve las últimas `n_sent` oraciones de un texto (o el texto completo si
    tiene menos oraciones que `n_sent`).\"\"\"
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\\s+", text) if p.strip()]
    return " ".join(parts[-n_sent:]) if parts else text

train_df["n_sentences"] = train_df["text"].apply(
    lambda t: len(re.split(r"(?<=[.!?])\\s+", t))
)
eval_df["n_sentences"] = eval_df["text"].apply(
    lambda t: len(re.split(r"(?<=[.!?])\\s+", t))
)
print("Oraciones por reseña — train:")
print(train_df["n_sentences"].describe())
print("\\nOraciones por reseña — eval (debe verse parecido, si no, el truco no generaliza):")
print(eval_df["n_sentences"].describe())
""")

md("""\
La distribución de número de oraciones es prácticamente idéntica entre `train.csv` y
`eval.csv` (media ≈3.9 en ambos), así que no hay riesgo de que esta transformación se
comporte distinto en el conjunto de Kaggle.
""")

code("""\
train_df["last1_clean"] = train_df["text"].apply(lambda t: clean_text(last_chunk(t, 1)))
eval_df["last1_clean"] = eval_df["text"].apply(lambda t: clean_text(last_chunk(t, 1)))

train_df[["text", "last1_clean", "label"]].sample(3, random_state=RANDOM_STATE)
""")

md("### 12.1 Comparación: texto completo vs. solo la última oración")

code("""\
X_train_last1 = train_df.loc[X_train_text.index, "last1_clean"]
X_val_last1 = train_df.loc[X_val_text.index, "last1_clean"]

comparacion_oraciones = []
for n_sent, nombre in [(None, "Texto completo"), (1, "Solo última oración"),
                        (2, "Últimas 2 oraciones")]:
    if n_sent is None:
        Xtr, Xva = X_train_text, X_val_text
    else:
        col = f"last{n_sent}_clean"
        if col not in train_df:
            train_df[col] = train_df["text"].apply(lambda t: clean_text(last_chunk(t, n_sent)))
        Xtr = train_df.loc[X_train_text.index, col]
        Xva = train_df.loc[X_val_text.index, col]

    pipe = Pipeline([
        ("vec", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("clf", LogisticRegression(max_iter=3000, C=3, random_state=RANDOM_STATE)),
    ])
    pipe.fit(Xtr, y_train)
    acc = accuracy_score(y_val, pipe.predict(Xva))
    comparacion_oraciones.append({"variante": nombre, "accuracy_val": acc})

pd.DataFrame(comparacion_oraciones)
""")

md("""\
El salto es enorme con **una sola oración** (la última), y agregar la penúltima
(`"Últimas 2 oraciones"`) en realidad empeora el resultado — confirma que el párrafo de
logística anterior funciona como ruido que diluye la señal, incluso cuando se incluye
solo parcialmente.

### 12.2 Afinando el modelo sobre la última oración

Repetimos una búsqueda de hiperparámetros enfocada (n-gramas, `min_df`, `sublinear_tf`,
`C`) ahora que el texto de entrada es mucho más corto y específico.
""")

code("""\
pipe_last1 = Pipeline([
    ("vec", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
])

param_grid_last1 = {
    "vec__ngram_range": [(1, 1), (1, 2), (1, 3)],
    "vec__min_df": [1, 2, 3],
    "vec__max_features": [None, 20000, 40000],
    "vec__sublinear_tf": [True, False],
    "clf__C": [0.3, 0.5, 0.7, 1, 2, 3],
}

search_last1 = RandomizedSearchCV(
    pipe_last1, param_distributions=param_grid_last1, n_iter=25, cv=cv,
    scoring="accuracy", random_state=RANDOM_STATE, n_jobs=-1, verbose=1,
)
search_last1.fit(X_train_last1, y_train)
print("Mejor accuracy (CV, train, última oración):", search_last1.best_score_)
print("Mejores hiperparámetros:", search_last1.best_params_)
""")

code("""\
last1_pipe = search_last1.best_estimator_
last1_val_preds = last1_pipe.predict(X_val_last1)
last1_val_acc = accuracy_score(y_val, last1_val_preds)
print(f"Accuracy en validación (última oración, tuneado): {last1_val_acc:.4f}")

fitted_models["TFIDF última oración (tuned) + LogisticRegression"] = last1_pipe
results_df = pd.concat([
    results_df,
    pd.DataFrame([{"modelo": "TFIDF última oración (tuned) + LogisticRegression",
                    "accuracy_val": last1_val_acc}]),
], ignore_index=True).sort_values("accuracy_val", ascending=False).reset_index(drop=True)
results_df.head(10)
""")

md("### 12.3 Evaluación del modelo ganador")

code("""\
print(classification_report(y_val, last1_val_preds, target_names=LABELS))
""")

code("""\
fig, ax = plt.subplots(figsize=(5.5, 5))
ConfusionMatrixDisplay.from_predictions(
    y_val, last1_val_preds, labels=LABELS, cmap="Blues", ax=ax, colorbar=False,
)
ax.set_title("Matriz de confusión — última oración (validación)")
plt.tight_layout()
plt.show()
""")

md("""\
`neutral` ahora se identifica casi perfectamente — tiene sentido, porque una reseña sin
una oración final claramente evaluativa (solo más detalles de logística) es, en sí
misma, una señal fuerte de neutralidad. La confusión que queda es sobre todo entre
`negativo` y `positivo`, justo donde el enunciado advertía que el matiz y la negación
son más difíciles de capturar con bolsa de palabras.

### 12.4 Entrenamiento final y envío a Kaggle
""")

code("""\
best_model_name = "TFIDF última oración (tuned) + LogisticRegression"
best_val_acc = last1_val_acc
best_pipe = last1_pipe
print(f"Modelo seleccionado para el envío: {best_model_name} "
      f"(accuracy validación = {best_val_acc:.4f})")

final_pipe_last1 = clone(best_pipe)
final_pipe_last1.fit(train_df["last1_clean"], train_df["label"])

eval_preds_last1 = final_pipe_last1.predict(eval_df["last1_clean"])
pd.Series(eval_preds_last1).value_counts(normalize=True).round(3)
""")

code("""\
submission_last1 = pd.DataFrame({"id": eval_df["id"], "answer": eval_preds_last1})
assert list(submission_last1.columns) == list(sample_submission.columns)
assert len(submission_last1) == len(sample_submission)
assert (submission_last1["id"].values == sample_submission["id"].values).all()
assert set(submission_last1["answer"].unique()) <= set(LABELS)

submission_path_last1 = SUBMISSIONS_DIR / "submission_v7_ultima_oracion.csv"
submission_last1.to_csv(submission_path_last1, index=False)

model_path_last1 = MODELS_DIR / "modelo_v7_ultima_oracion.joblib"
joblib.dump(final_pipe_last1, model_path_last1)

print("Guardado:", submission_path_last1)
print("Guardado:", model_path_last1)
submission_last1.head()
""")

md("""\
### 12.5 Qué significa este resultado

Pasar de ~0.73 a ~0.86 de accuracy en validación **no vino de un modelo más complejo**,
sino de una transformación de los datos basada en entender la estructura de la
plantilla con la que están escritas las reseñas. Esto es exactamente el tipo de
"feature engineering" que la Parte 1 del curso busca que se explore antes de pasar a
deep learning en la Parte 2.

**Limitaciones a tener en cuenta:**

- Este truco asume que la oración final siempre contiene la opinión. Si el patrón de
  generación de las reseñas cambia (por ejemplo, reseñas más cortas o con la opinión al
  principio), el modelo se vería afectado. Vale la pena revisar casos donde falle.
- Seguimos sin capturar bien negaciones/contrastes *dentro* de la última oración misma
  (ver matriz de confusión, sección 12.3) — ahí es donde modelos secuenciales (Parte 2)
  deberían ayudar más.
- Ideas para seguir mejorando desde acá: probar con las últimas 1-2 oraciones pero
  separadas como dos campos de un `FeatureUnion` en lugar de concatenadas; detectar la
  oración que sigue al *último* conector contrastivo en vez de simplemente "la última";
  o combinar esta señal con alguna característica agregada del resto del texto
  únicamente para reforzar la detección de `neutral`.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

with open("notebooks/parte1_ml_clasico.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook generado.")
