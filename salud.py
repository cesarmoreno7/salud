import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split

# Cargar datos
def load_data():
    file_path = "healthcare_dataset.csv"  # Asegúrate de subir el archivo en Streamlit
    df = pd.read_csv(file_path)
    df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
    df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
    return df

df = load_data()

# Sidebar
st.sidebar.title("Menú Principal")
start_date = st.sidebar.date_input("Fecha Inicio", df['Date of Admission'].min())
end_date = st.sidebar.date_input("Fecha Final", df['Date of Admission'].max())
option = st.sidebar.radio("Seleccione una opción", ["Análisis exploratorio", "Visualización de datos", "Predicción"])

# Filtrar datos por fecha
df_filtered = df[(df['Date of Admission'] >= pd.Timestamp(start_date)) & (df['Date of Admission'] <= pd.Timestamp(end_date))]

# Análisis exploratorio
if option == "Análisis exploratorio":
    st.title("Análisis Exploratorio de Datos")
    st.write(df_filtered.describe())
    
    chart_type = st.selectbox("Seleccione el tipo de gráfico", ["Histograma", "Boxplot", "Dispersión"])
    fig, ax = plt.subplots()
    
    if chart_type == "Histograma":
        sns.histplot(df_filtered['Billing Amount'], bins=20, kde=True, ax=ax)
        st.write("**Interpretación:** Un histograma permite observar la distribución de la facturación. Si es simétrico, indica estabilidad en los valores; si está sesgado, puede haber valores extremos influyentes.")
        st.write("**Acción de mejora:** Analizar la causa de valores extremos y ajustar políticas de precios o facturación.")
    elif chart_type == "Boxplot":
        sns.boxplot(x=df_filtered['Billing Amount'], ax=ax)
        st.write("**Interpretación:** Un boxplot muestra la dispersión de los montos facturados. La presencia de muchos outliers puede indicar facturación irregular.")
        st.write("**Acción de mejora:** Revisar las políticas de facturación para reducir variaciones extremas y mejorar la estabilidad financiera.")
    elif chart_type == "Dispersión":
        sns.scatterplot(x=df_filtered.index, y=df_filtered['Billing Amount'], ax=ax)
        st.write("**Interpretación:** Un gráfico de dispersión ayuda a identificar tendencias o patrones estacionales en la facturación.")
        st.write("**Acción de mejora:** Implementar estrategias de promoción o ajuste de servicios en períodos de baja facturación.")
    
    st.pyplot(fig)

# Visualización de datos
elif option == "Visualización de datos":
    st.title("Visualización de Datos")
    category = st.selectbox("Seleccione una categoría", ["Medical Condition", "Hospital", "Insurance Provider"])
    chart_type = st.selectbox("Seleccione el tipo de gráfico", ["Barras", "Pie Chart"])
    
    fig, ax = plt.subplots()
    
    if chart_type == "Barras":
        sns.countplot(data=df_filtered, x=category, ax=ax)
        plt.xticks(rotation=45)
        st.write(f"**Interpretación:** Este gráfico muestra la frecuencia de cada {category}. Una alta concentración en ciertas categorías indica dependencia de ciertos servicios o proveedores.")
        st.write("**Acción de mejora:** Diversificar los proveedores o servicios para mitigar riesgos y mejorar la oferta.")
    elif chart_type == "Pie Chart":
        df_filtered[category].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
        st.write(f"**Interpretación:** El gráfico circular permite visualizar la proporción de cada {category}. Si un grupo domina, puede representar una dependencia fuerte.")
        st.write("**Acción de mejora:** Evaluar estrategias para equilibrar la distribución y mejorar la equidad en la atención médica.")
    
    st.pyplot(fig)

# Predicción de facturación
elif option == "Predicción":
    st.title("Predicción de Facturación")
    days_to_predict = st.selectbox("Seleccione días a predecir", [7, 15, 30])
    
    # Modelo simple de regresión lineal
    df_filtered['Days'] = (df_filtered['Date of Admission'] - df_filtered['Date of Admission'].min()).dt.days
    X = df_filtered[['Days']]
    y = df_filtered['Billing Amount']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    future_days = np.array([df_filtered['Days'].max() + i for i in range(1, days_to_predict+1)]).reshape(-1, 1)
    future_predictions = model.predict(future_days)
    
    future_df = pd.DataFrame({"Días Futuros": future_days.flatten(), "Facturación Predicha": future_predictions})
    st.write(future_df)
    
    st.write("**Interpretación:** La proyección indica la tendencia futura de la facturación en base a los datos históricos. Desviaciones significativas pueden sugerir cambios en la demanda o errores en la predicción.")
    st.write("**Acción de mejora:** Ajustar estrategias de ventas y marketing para mejorar los ingresos en función de la tendencia prevista.")
