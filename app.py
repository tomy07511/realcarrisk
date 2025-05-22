# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

# Configuración de la página
st.set_page_config(page_title="Clasificador de Riesgo Vehicular", layout="wide")
st.title("Clasificador CarrRisk")

# Cargar el modelo
@st.cache_resource
def load_model():
    with open('modelo-clas-tree-knn-nn.pkl', 'rb') as file:
        return pickle.load(file)

model_Tree, model_Knn, model_NN, labelencoder, variables, min_max_scaler = load_model()

# Función para clasificar el riesgo
def classify_risk(prediction):
    return 'Alto Riesgo' if prediction == 'high' else 'Bajo Riesgo'

# Sidebar para entrada de datos
st.sidebar.header("Parámetros del Vehículo")

# Obtener opciones específicas del modelo
TIPOS_VEHICULO = ['combi', 'sport', 'family', 'minivan']

def get_input():
    # Datos del vehículo
    edad = st.sidebar.slider("Edad del conductor", 18, 80, 30)
    tipo = st.sidebar.selectbox("Tipo de vehículo", TIPOS_VEHICULO)
    
    return {
        'age': edad,
        'cartype': tipo
    }

# Procesamiento de datos
def prepare_data(input_data):
    data = pd.DataFrame([input_data])
    
    # One-hot encoding
    data = pd.get_dummies(data, columns=['cartype'], drop_first=False)
    
    # Asegurar todas las columnas del modelo
    missing_cols = set(variables) - set(data.columns)
    for col in missing_cols:
        data[col] = 0
    
    return data[variables]

# Interfaz principal
user_input = get_input()

# Selección del modelo
model_option = st.sidebar.radio("Modelo a utilizar", 
                               ['Árbol de Decisión', 'K-Vecinos', 'Red Neuronal'],
                               index=0)

if st.sidebar.button(" Calcular Riesgo"):
    data_prep = prepare_data(user_input)
    
    # Predicción según modelo seleccionado
    if model_option == 'Árbol de Decisión':
        prediction = model_Tree.predict(data_prep)
    elif model_option == 'K-Vecinos':
        data_knn = data_prep.copy()
        data_knn[['age']] = min_max_scaler.transform(data_knn[['age']])
        prediction = model_Knn.predict(data_knn)
    else:
        data_nn = data_prep.copy()
        data_nn[['age']] = min_max_scaler.transform(data_nn[['age']])
        prediction = model_NN.predict(data_nn)
    
    # Decodificar la predicción
    risk_level = labelencoder.inverse_transform(prediction)[0]
    
    # Mostrar resultados
    st.success(f"Predicción de riesgo: {classify_risk(risk_level)}")
    
    # Detalles técnicos
    with st.expander("Detalles técnicos"):
        st.write("**Datos de entrada:**")
        st.json(user_input)
        st.write("**Datos preparados para el modelo:**")
        st.dataframe(data_prep)

# Sección para carga de archivos
st.header("Opción Avanzada: Carga Masiva")
uploaded_file = st.file_uploader("Sube un CSV con múltiples registros", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Datos cargados:", df.head())
    
    if st.button("🔮 Predecir para todos"):
        # Preparar datos
        df_prep = pd.get_dummies(df, columns=['cartype'], drop_first=False)
        
        # Asegurar columnas del modelo
        missing_cols = set(variables) - set(df_prep.columns)
        for col in missing_cols:
            df_prep[col] = 0
        
        # Predicciones
        df['Prediccion_Arbol'] = [classify_risk(x) for x in labelencoder.inverse_transform(model_Tree.predict(df_prep[variables]))]
        
        df_knn = df_prep.copy()
        df_knn[['age']] = min_max_scaler.transform(df_knn[['age']])
        df['Prediccion_KNN'] = [classify_risk(x) for x in labelencoder.inverse_transform(model_Knn.predict(df_knn[variables]))]
        
        df_nn = df_prep.copy()
        df_nn[['age']] = min_max_scaler.transform(df_nn[['age']])
        df['Prediccion_NN'] = [classify_risk(x) for x in labelencoder.inverse_transform(model_NN.predict(df_nn[variables]))]
        
        st.success(f"Predicciones completadas para {len(df)} registros")
        st.dataframe(df)
        
        # Exportar resultados
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Descargar resultados",
            csv,
            "resultados_riesgo.csv",
            "text/csv"
        )