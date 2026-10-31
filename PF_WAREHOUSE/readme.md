# 📦 PF_WAREHOUSE: Sistema MLOps de Predicción de Demanda (Walmart)

Proyecto MLOps integral enfocado en el procesamiento de datos históricos, entrenamiento optimizado de modelos de Machine Learning, despliegue automatizado de una API en la nube y análisis exploratorio de datos (EDA) para la predicción de ventas semanales en tiendas minoristas.

---

## 🚀 Enlaces del Proyecto
* **API en Producción (Swagger UI):** [https://pf-warehouse-api.onrender.com/docs](https://pf-warehouse-api.onrender.com/docs)
* **Estado del Despliegue:** Activo en Render (Cloud)

---

## 🛠️ Arquitectura y Stack Tecnológico
* **Lenguaje:** Python 3.10+
* **Backend / API:** FastAPI, Uvicorn, Pydantic
* **Machine Learning:** Scikit-Learn (RandomForestRegressor), Pandas, NumPy
* **Despliegue y MLOps:** Render (Cloud Hosting), Git / GitHub
* **Visualización y EDA:** Matplotlib, Seaborn

---

## 📊 Estructura del Repositorio
```text
PF_WAREHOUSE/
│
├── data/
│   └── train.csv               # Dataset histórico (2010 - 2012)
├── models/
│   └── model.joblib            # Modelo entrenado serializado
├── reports/
│   └── figures/                # Gráficos del Análisis Exploratorio (EDA)
│       ├── distribucion_ventas.png
│       ├── ventas_por_mes.png
│       └── impacto_festivos.png
├── src/
│   ├── api.py                  # Código de la API FastAPI y ciclo de vida (Lifespan)
│   ├── train.py                # Script de preprocesamiento y entrenamiento optimizado
│   └── eda.py                  # Script de análisis exploratorio de datos
├── requirements.txt            # Dependencias del proyecto
└── README.md                   # Documentación principal

📈 Análisis Exploratorio de Datos (EDA)
Se realizó un análisis estadístico y visual sobre el comportamiento histórico de las ventas (Weekly_Sales), identificando patrones clave:

Distribución de Ventas: Comportamiento de las ventas semanales por departamento y tienda.

Estacionalidad Mensual: Identificación de los meses con mayor volumen de transacciones a lo largo del año.

Impacto de Feriados: Evaluación del efecto de los días festivos (IsHoliday) frente a semanas regulares.

(Los gráficos generados se encuentran en la carpeta reports/figures/).

⚙️ Funcionamiento de la API
La API cuenta con un endpoint principal (/predict) que recibe los parámetros clave del negocio y retorna la predicción de ventas semanales optimizada a 2 decimales:

Parámetros de Entrada:

store: Número de tienda.

dept: Número de departamento.

is_holiday: Indicador de día festivo (0 o 1).

year, month, day: Fecha a consultar dentro del rango histórico del modelo.

🚀 Instalación y Ejecución Local (Opcional)
Si deseas clonar y ejecutar el proyecto en tu entorno local:

Clonar el repositorio:

Bash
git clone [https://github.com/tu-usuario/PF_WAREHOUSE.git](https://github.com/tu-usuario/PF_WAREHOUSE.git)
cd PF_WAREHOUSE
Crear y activar el entorno virtual:

Bash
python -m venv .venv
# En Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
Instalar dependencias:

Bash
pip install -r requirements.txt
Ejecutar la API localmente:

Bash
uvicorn src.api:app --reload

## 🎨 Interfaz Gráfica Local (Gradio)
Además de la API en Render, el proyecto incluye una interfaz web interactiva desarrollada con **Gradio** para realizar pruebas visuales locales de manera intuitiva mediante controles deslizantes y selectores.

* **Ejecutar la interfaz de Gradio localmente:**
  ```bash
  python src/app_gradio.py
Esto abrirá una aplicación web interactiva en tu navegador (por defecto en http://127.0.0.1:7860).