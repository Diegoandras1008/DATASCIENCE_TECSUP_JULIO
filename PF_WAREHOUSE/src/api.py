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
from fastapi import FastAPI, Query
import joblib
import pandas as pd
import os
from src.train import ejecutar_entrenamiento

# Variable global para el modelo
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    # 1. Si el modelo no existe, lo entrenamos automáticamente al arrancar
    if not os.path.exists("models/model.joblib"):
        print(">>> Modelo no encontrado. Entrenando automáticamente en el servidor...")
        ejecutar_entrenamiento()
    
    # 2. Cargamos el modelo en memoria para la API
    print(">>> Cargando el modelo entrenado...")
    model = joblib.load("models/model.joblib")
    yield

app = FastAPI(
    title="API Predicción de Demanda - Walmart MLOps",
    version="1.0",
    lifespan=lifespan
)

@app.get("/")
def read_root():
    return {"mensaje": "¡La API de predicción de demanda está activa y funcionando! 🚀"}

@app.post("/predict", summary="Predecir ventas semanales")
def predict_demand(
    store: int = Query(..., description="Número de tienda (Store)", example=10),
    dept: int = Query(..., description="Número de departamento (Dept)", example=2),
    is_holiday: int = Query(..., description="Es día festivo (0 = No, 1 = Sí)", example=0),
    year: int = Query(..., description="Año dentro del rango histórico (ej. 2012)", example=2012),
    month: int = Query(..., description="Mes (1-12)", example=9),
    day: int = Query(..., description="Día del mes (1-31)", example=21)
):
    input_df = pd.DataFrame([{
        'Store': store,
        'Dept': dept,
        'IsHoliday': is_holiday,
        'Year': year,
        'Month': month,
        'Day': day
    }])
    
    prediction = model.predict(input_df)
    
    # Redondear a 2 decimales para una presentación limpia
    predicted_sales = round(float(prediction[0]), 2)
    
    return {
        "Store": store,
        "Dept": dept,
        "Predicted_Weekly_Sales": predicted_sales
    }