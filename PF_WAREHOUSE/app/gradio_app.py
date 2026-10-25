import gradio as gr
import requests

# URL de la API de FastAPI en ejecución
API_URL = "http://127.0.0.1:8000/predict"


def predecir_demanda(store, dept, is_holiday, year, month, week):
    payload = {
        "Store": int(store),
        "Dept": int(dept),
        "IsHoliday": int(is_holiday),
        "Year": int(year),
        "Month": int(month),
        "Week": int(week),
    }

    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            resultado = response.json()
            return f"Ventas Semanales Estimadas: ${resultado['prediccion_ventas_semanales']:,.2f}"
        else:
            return f"Error en la API: {response.json().get('detail', 'Error desconocido')}"
    except Exception as e:
        return f"No se pudo conectar con la API de FastAPI. Asegúrate de que esté encendida. Detalles: {str(e)}"


# Construir la interfaz visual con Gradio
demo = gr.Interface(
    fn=predecir_demanda,
    inputs=[
        gr.Number(label="Tienda (Store)", value=1),
        gr.Number(label="Departamento (Dept)", value=1),
        gr.Dropdown(
            choices=[0, 1], label="Es Feriado (0 = No, 1 = Sí)", value=0
        ),
        gr.Number(label="Año (Year)", value=2012),
        gr.Number(label="Mes (Month)", value=12),
        gr.Number(label="Semana del año (Week)", value=50),
    ],
    outputs=gr.Text(label="Predicción de Inventario"),
    title="PF_WAREHOUSE - Predicción de Demanda de Inventario (Walmart)",
    description="Ingrese los parámetros operativos para calcular la demanda semanal estimada mediante el modelo MLOps.",
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)