import pandas as pd
from .config import RAW_DATA_PATH  # Usamos un punto (.) para indicar que está en la misma carpeta


def cargar_y_validar_datos(filepath: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Carga el dataset de Walmart y realiza validaciones básicas de columnas y nulos."""
    if not filepath.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de datos en: {filepath}. Asegúrate de colocar train.csv en la carpeta data/"
        )

    df = pd.read_csv(filepath)

    # 1. Validación de columnas obligatorias requeridas
    cols_requeridas = ["Store", "Dept", "Date", "Weekly_Sales", "IsHoliday"]
    for col in cols_requeridas:
        if col not in df.columns:
            raise ValueError(f"Falta la columna obligatoria en el dataset: {col}")

    # 2. Limpieza de nulos en la variable objetivo
    df = df.dropna(subset=["Weekly_Sales"]).copy()

    # 3. Conversión de tipos y limpieza de rangos no válidos
    df["IsHoliday"] = df["IsHoliday"].astype(int)

    # Filtramos ventas negativas si las hubiera
    if (df["Weekly_Sales"] < 0).any():
        df = df[df["Weekly_Sales"] >= 0]

    return df


def preparar_features(df: pd.DataFrame):
    """Extrae las características temporales y numéricas para el modelo."""
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)

    # Features seleccionadas para predecir la demanda de inventario
    X = df[["Store", "Dept", "IsHoliday", "Year", "Month", "Week"]]
    y = df["Weekly_Sales"]

    return X, y