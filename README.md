# Proyecto — Clasificación de Sentimiento (Aprendizaje de Máquina 2026-20)

## Parte 1: Machine Learning Clásico

Primera iteración del proyecto de competencia en Kaggle: clasificar reseñas de
producto en español como `negativo`, `neutral` o `positivo`, usando únicamente
técnicas clásicas de ML (bag-of-words / TF-IDF + scikit-learn), sin redes
neuronales ni embeddings preentrenados, tal como exige el enunciado para esta
primera etapa.

## Contenido de la carpeta

```
Proyectos202620/
├── data/
│   ├── train.csv              # 12,000 reseñas etiquetadas
│   ├── eval.csv                # 3,000 reseñas a predecir (test de Kaggle)
│   └── sample_submission.csv   # formato exacto que espera Kaggle
├── notebooks/
│   ├── parte1_ml_clasico.ipynb # notebook principal (EDA + modelado + submission)
│   └── parte1_ml_clasico.html  # versión HTML del notebook ya ejecutado, para
│                                  verlo rápido sin abrir Jupyter
├── models/
│   └── modelo_tfidf_logisticregression.joblib  # modelo final (joblib)
├── submissions/
│   └── submission_v1_tfidf_logisticregression.csv  # listo para subir a Kaggle
└── build_notebook.py           # script que genera el notebook (útil si quieren
                                    modificar la estructura y regenerarlo)
```

## Resultado de esta primera iteración

Se compararon 8 configuraciones (Dummy, Bag-of-Words + Naive Bayes, TF-IDF +
Naive Bayes/Logistic Regression/LinearSVC/SGD, con y sin balanceo de clases, y
una versión con `RandomizedSearchCV`). El mejor resultado en validación local
(15% de train.csv, estratificado) fue:

- **Modelo:** TF-IDF (unigramas + bigramas) + `LogisticRegression`
- **Accuracy en validación:** ≈ 0.734
- **Piso de referencia (Dummy):** ≈ 0.351

El notebook documenta explícitamente que el modelo tuneado con
`RandomizedSearchCV` **no** superó al modelo por defecto en la validación
(0.725 vs. 0.734), así que el notebook compara todos los candidatos y elige
objetivamente el mejor en vez de asumir que el tuning siempre gana — vale la
pena tenerlo en cuenta para no repetir el mismo error en las próximas
iteraciones.

## Qué falta / próximos pasos

1. **Subir `submission_v1_tfidf_logisticregression.csv` a Kaggle** (necesito
   el link de invitación de la competencia, que según el enunciado se publica
   en Bloque Neón) — ya deberías tener superávit sobre el baseline del curso
   con este primer envío.
2. La rúbrica pide **al menos 5 envíos distintos** mostrando mejora
   progresiva. El notebook (sección 10) trae una lista concreta de ideas para
   las siguientes iteraciones: character n-grams, features léxicas hechas a
   mano, oversampling de la clase `neutral`, ensambles, etc.
3. Documentar en el notebook (o en una bitácora aparte) quién del equipo hizo
   qué en cada envío, para sustentar el aporte individual que pide la
   rúbrica.
4. Cuando avancen a la Parte 2 (deep learning + aumentación de datos), esta
   misma carpeta `Proyectos202620` es el lugar natural para agregar esa
   segunda etapa.

## Notas

- La carpeta `Proyectos` (sin 202620) tiene material de un semestre anterior
  del curso, pero corresponde a una competencia distinta (clasificación de
  década de textos históricos, con deep learning pesado) — no se reutilizó
  directamente porque el problema y las reglas de esta Parte 1 son distintos,
  aunque la disciplina de documentar cada iteración sí vale la pena mantenerla.
