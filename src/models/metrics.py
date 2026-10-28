"""Metricas de evaluacion para clasificacion binaria de video.

Se separan de train.py para poder reutilizarlas desde evaluate.py y la API
y para poder testearlas sin depender de un modelo cargado.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str] | None = None,
) -> dict:
    """Calcula metricas por clase y globales para un lote de predicciones.

    Args:
        y_true: etiquetas verdaderas, enteros en {0, 1}.
        y_pred: etiquetas predichas por el modelo (argmax de los logits).
        class_names: nombres de las clases en orden (indice 0 primero). Si
            es None, se usa `["clase_0", "clase_1"]`.

    Returns:
        Diccionario con:
            - accuracy: float en [0, 1].
            - f1_macro: float en [0, 1], promedio simple de los F1 por clase.
              Esta es la metrica que elige el mejor checkpoint: con clases
              desbalanceadas (p.ej. 60/40), accuracy sube siempre pero F1
              macro baja si el modelo ignora la clase chica.
            - f1_per_class: dict `nombre -> f1`.
            - precision_per_class: dict `nombre -> precision`.
            - recall_per_class: dict `nombre -> recall`.
            - confusion_matrix: lista de listas 2x2, filas = verdadero,
              columnas = predicho (convencion de sklearn).
    """
    # sklearn trabaja con arrays contiguos; convertimos de una vez para
    # evitar pedir np.asarray en cada metrica.
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    labels = [0, 1]
    if class_names is None:
        class_names = ["clase_0", "clase_1"]
    if len(class_names) != 2:
        raise ValueError(
            f"class_names debe tener 2 nombres, llegaron {len(class_names)}"
        )

    # zero_division=0: con un lote muy chico (p.ej. smoke test con 5 videos)
    # puede que una clase no aparezca; sin este flag sklearn imprime warnings.
    f1_per = f1_score(y_true_arr, y_pred_arr, labels=labels, average=None,
                      zero_division=0)
    precision_per = precision_score(y_true_arr, y_pred_arr, labels=labels,
                                    average=None, zero_division=0)
    recall_per = recall_score(y_true_arr, y_pred_arr, labels=labels,
                              average=None, zero_division=0)
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)

    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "f1_macro": float(f1_score(y_true_arr, y_pred_arr, labels=labels,
                                   average="macro", zero_division=0)),
        "f1_per_class": {n: float(v) for n, v in zip(class_names, f1_per)},
        "precision_per_class": {n: float(v) for n, v in zip(class_names, precision_per)},
        "recall_per_class": {n: float(v) for n, v in zip(class_names, recall_per)},
        "confusion_matrix": cm.tolist(),
    }
