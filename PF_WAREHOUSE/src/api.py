import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import METRICS_PATH, MODEL_PATH

app = FastAPI(
    title="PF_WAREHOUSE - Predicción de Demanda",
    version="1.0.0",
    description="API MLOps para predecir las ventas semanales de inventario en tiendas Walmart.",
)

# Cargar el modelo entrenado al iniciar la API
try:
    model = joblib.load(MODEL_PATH)
except Exception:
    model = None

# Cargar las métricas del modelo
try:
    with open(METRICS_PATH, "r") as f:
        metrics_data = json.load(f)
except Exception:
    metrics_data = {}


# Esquema Pydantic para validar los datos de entrada de la API
class ItemDemanda(BaseModel):
    Store: int
    Dept: int
    IsHoliday: int
    Year: int
    Month: int
    Week: int


@app.get("/health")
def health_check():
    """Verifica el estado del servicio y si el modelo está cargado."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/model-info")
def model_info():
    """Devuelve las métricas y parámetros del modelo entrenado."""
    if not metrics_data:
        raise HTTPException(
            status_code=404, detail="No se encontró la ficha de métricas."
        )
    return metrics_data


@app.post("/predict")
def predict(item: ItemDemanda):
    """Realiza una predicción de la demanda semanal de inventario."""
    if model is None:
        raise HTTPException(
            status_code=500, detail="El modelo no está cargado en el servidor."
        )

    # Crear DataFrame con los datos de entrada validados
    input_data = pd.DataFrame(
        [
            {
                "Store": item.Store,
                "Dept": item.Dept,
                "IsHoliday": item.IsHoliday,
                "Year": item.Year,
                "Month": item.Month,
                "Week": item.Week,
            }
        ]
    )

    try:
        prediction = model.predict(input_data)[0]
        return {"prediccion_ventas_semanales": round(float(prediction), 2)}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error durante la inferencia: {str(e)}"
        )

from contextlib import asynccontextmanager
from fastapi import FastAPI
import os
import joblib
from src.train import ejecutar_entrenamiento  # Importa tu función de entrenamiento

# Definimos el evento de inicio de la aplicación
@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = "models/model.joblib"
    
    # Si el modelo no existe (porque Git lo ignoró), lo entrenamos al arrancar
    if not os.path.exists(model_path):
        print(">>> Modelo no encontrado. Entrenando automáticamente en el servidor...")
        ejecutar_entrenamiento()
    else:
        print(">>> Modelo cargado correctamente desde el disco.")
        
    yield
    # Código de limpieza al apagar (si se requiere)

app = FastAPI(
    title="API Predicción de Demanda - Walmart MLOps",
    version="1.0",
    lifespan=lifespan
)

# Cargar el modelo en memoria para las predicciones
MODEL_PATH = "models/model.joblib"
model = None

# ... (tus rutas de FastAPI y endpoints de predicción continúan aquí)
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(
    title="API Predicción de Demanda - Walmart MLOps",
    version="1.0"
)

# Cargar el modelo entrenado (puedes usar tu ciclo de vida 'lifespan' o cargarlo aquí)
model = joblib.load("models/model.joblib")

@app.get("/")
def read_root():
    return {"mensaje": "¡La API de predicción de demanda está activa y funcionando! 🚀"}

@app.post("/predict")
def predict_demand(store: int, dept: int, is_holiday: int, year: int, month: int, day: int):
    # Preparar los datos de entrada en el mismo orden que el entrenamiento
    input_data = pd.DataFrame([{
        'Store': store,
        'Dept': dept,
        'IsHoliday': is_holiday,
        'Year': year,
        'Month': month,
        'Day': day
    }])
    
    # Realizar la predicción con el modelo
    prediction = model.predict(input_data)
    
    return {
        "Store": store,
        "Dept": dept,
        "Predicted_Weekly_Sales": float(prediction[0])
    }