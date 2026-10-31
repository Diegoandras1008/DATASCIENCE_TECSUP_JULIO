import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def realizar_eda():
    print("==================================================")
    print("   INICIANDO ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
    print("==================================================")
    
    # 1. Cargar el dataset histórico
    if not os.path.exists("data/train.csv"):
        print("Error: No se encuentra el archivo data/train.csv")
        return
        
    df = pd.read_csv("data/train.csv")
    
    # 2. Resumen estructural
    print(f"\n[Dimensiones del Dataset]")
    print(f"• Filas: {df.shape[0]:,}")
    print(f"• Columnas: {df.shape[1]}")
    
    print(f"\n[Valores Nulos por Columna]")
    print(df.isnull().sum())
    
    # 3. Estadísticas de la variable objetivo (Weekly_Sales)
    print(f"\n[Estadísticas de la Variable Objetivo: Weekly_Sales]")
    print(df['Weekly_Sales'].describe())
    
    # 4. Crear directorio para guardar las figuras del reporte
    os.makedirs("reports/figures", exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # --- GRÁFICO 1: Distribución de las Ventas Semanales ---
    print("\nGenerando gráfico de distribución de ventas...")
    plt.figure(figsize=(9, 5))
    sns.histplot(df['Weekly_Sales'], bins=50, kde=True, color='teal')
    plt.title('Distribución de Ventas Semanales (Weekly_Sales)', fontsize=12, fontweight='bold')
    plt.xlabel('Ventas Semanales')
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    plt.savefig("reports/figures/distribucion_ventas.png", dpi=300)
    plt.close()
    
    # --- GRÁFICO 2: Ventas promedio por mes (Estacionalidad) ---
    if 'Date' in df.columns:
        print("Generando gráfico de estacionalidad mensual...")
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        monthly_sales = df.groupby('Month')['Weekly_Sales'].mean().reset_index()
        
        plt.figure(figsize=(9, 5))
        sns.barplot(x='Month', y='Weekly_Sales', data=monthly_sales, palette='Blues_d')
        plt.title('Ventas Promedio por Mes (Estacionalidad)', fontsize=12, fontweight='bold')
        plt.xlabel('Mes del Año')
        plt.ylabel('Ventas Promedio Semanales')
        plt.tight_layout()
        plt.savefig("reports/figures/ventas_por_mes.png", dpi=300)
        plt.close()

    # --- GRÁFICO 3: Impacto de días festivos (IsHoliday) ---
    if 'IsHoliday' in df.columns:
        print("Generando gráfico de impacto por días festivos...")
        plt.figure(figsize=(7, 5))
        sns.boxplot(x='IsHoliday', y='Weekly_Sales', data=df, palette='Set2')
        plt.title('Impacto de Días Festivos en las Ventas', fontsize=12, fontweight='bold')
        plt.xlabel('Es Feriado (0 = No, 1 = Sí)')
        plt.ylabel('Ventas Semanales')
        plt.tight_layout()
        plt.savefig("reports/figures/impacto_festivos.png", dpi=300)
        plt.close()

    print("\n¡EDA finalizado con éxito!")
    print("Gráficos guardados en la carpeta: reports/figures/")
    print("==================================================")

if __name__ == "__main__":
    realizar_eda()