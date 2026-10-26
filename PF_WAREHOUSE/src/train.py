import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from .config import METRICS_PATH, MODEL_PATH, QUALITY_GATE_RMSE, SEED, TEST_SIZE
from .data import cargar_y_validar_datos, preparar_features


def entrenar_modelo():
    print("1. Cargando y validando datos...")
    df = cargar_y_validar_datos()
    X, y = preparar_features(df)

    print("2. Dividiendo datos (Train/Test) con semilla fija...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    print("3. Entrenando el modelo de Machine Learning...")
    model = RandomForestRegressor(n_estimators=50, random_state=SEED, n_jobs=-1)
    model.fit(X_train, y_train)

    print("4. Evaluando el modelo...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"-> RMSE obtenido en Test: {rmse:.2f}")

    # Puerta de calidad (Quality Gate) exigida por la rúbrica
    if rmse > QUALITY_GATE_RMSE:
        raise ValueError(
            f"El modelo no superó la puerta de calidad. RMSE actual: {rmse:.2f} (Umbral máximo permitido: {QUALITY_GATE_RMSE})"
        )
    print("¡El modelo superó la puerta de calidad con éxito!")

    # Guardar artefactos
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metrics = {
        "version": "1.0.0",
        "fecha": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metricas": {"rmse": round(float(rmse), 2)},
        "parametros": {"n_estimators": 50, "test_size": TEST_SIZE, "seed": SEED},
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Modelo guardado en {MODEL_PATH}")
    print(f"Métricas guardadas en {METRICS_PATH}")


if __name__ == "__main__":
    entrenar_modelo()

def ejecutar_entrenamiento():
    # Tu código actual de lectura de datos, preprocesamiento y entrenamiento
    # ...
    # Y al final, el guardado del modelo:
    import os
    import joblib
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.joblib")
    print("¡Modelo entrenado y guardado con éxito!")

if __name__ == "__main__":
    ejecutar_entrenamiento()

def ejecutar_entrenamiento():
    print("3. Entrenando el modelo de Machine Learning...")
    model = RandomForestRegressor(n_estimators=50, random_state=SEED, n_jobs=-1)
    model.fit(X_train, y_train)
    
    import os
    import joblib
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.joblib")
    print("¡Modelo entrenado y guardado con éxito!")