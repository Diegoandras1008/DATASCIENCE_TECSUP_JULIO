import joblib
import pandas as pd
import gradio as gr

# Cargar el modelo entrenado
model = joblib.load("models/model.joblib")

def predecir_ventas(store, dept, is_holiday, year, month, day):
    # Crear el DataFrame con los nombres de columnas exactos que espera el modelo
    input_df = pd.DataFrame([{
        'Store': int(store),
        'Dept': int(dept),
        'IsHoliday': int(is_holiday),
        'Year': int(year),
        'Month': int(month),
        'Day': int(day)
    }])
    
    # Realizar la predicción
    prediction = model.predict(input_df)
    predicted_sales = round(float(prediction[0]), 2)
    
    return f"${predicted_sales:,.2f}"

# Diseñar la interfaz visual con componentes limpios
demo = gr.Interface(
    fn=predecir_ventas,
    inputs=[
        gr.Number(label="Número de Tienda (Store)", value=10),
        gr.Number(label="Número de Departamento (Dept)", value=2),
        gr.Dropdown(choices=[0, 1], label="¿Es Feriado? (0 = No, 1 = Sí)", value=0),
        gr.Number(label="Año", value=2012),
        gr.Slider(minimum=1, maximum=12, step=1, label="Mes", value=9),
        gr.Slider(minimum=1, maximum=31, step=1, label="Día", value=21)
    ],
    outputs=gr.Textbox(label="Ventas Semanales Predichas ($)"),
    title="Predicción de Demanda - Walmart MLOps",
    description="Ingresa los parámetros logísticos para calcular la estimación de ventas semanales del almacén."
)

if __name__ == "__main__":
    demo.launch()