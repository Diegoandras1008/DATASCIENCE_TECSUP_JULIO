from fastapi.testclient import TestClient
from src.api import app
from src.config import RAW_DATA_PATH
from src.data import cargar_y_validar_datos

client = TestClient(app)


def test_archivo_datos_existe():
    """Verifica que el dataset de Walmart exista en la ruta configurada."""
    assert RAW_DATA_PATH.exists(), f"No se encontró el archivo en {RAW_DATA_PATH}"


def test_carga_datos():
    """Verifica que el dataset se cargue correctamente como un DataFrame y no esté vacío."""
    df = cargar_y_validar_datos()
    assert not df.empty, "El DataFrame cargado está vacío."
    assert "Weekly_Sales" in df.columns, (
        "La columna objetivo 'Weekly_Sales' no está presente."
    )


def test_api_health():
    """Verifica que el endpoint de salud de la API responda correctamente."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_api_predict():
    """Verifica que el endpoint de predicción devuelva un resultado válido."""
    payload = {
        "Store": 1,
        "Dept": 1,
        "IsHoliday": 0,
        "Year": 2012,
        "Month": 12,
        "Week": 50,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediccion_ventas_semanales" in data
    assert isinstance(data["prediccion_ventas_semanales"], float)