import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.title("Predicción de Aprobación de Curso")
st.write("Esta aplicación procesa las variables de entrada y realiza predicciones utilizando un modelo de Bagging pre-entrenado.")

# Función para cargar artefactos buscando primero de forma relativa para GitHub y luego en /content/ como respaldo
def cargar_artefacto(nombre_archivo):
    if os.path.exists(nombre_archivo):
        return joblib.load(nombre_archivo)
    elif os.path.exists(os.path.join('/content', nombre_archivo)):
        return joblib.load(os.path.join('/content', nombre_archivo))
    else:
        raise FileNotFoundError(f"No se encontró el archivo: {nombre_archivo}")

try:
    columnas_one_hot = cargar_artefacto('one_hot_columns.joblib')
    scaler = cargar_artefacto('min_max_scaler.joblib')
    modelo_bagging = cargar_artefacto('bagging_optimizado.joblib')
    st.sidebar.success("Modelos y escaladores cargados correctamente.")
except Exception as e:
    st.sidebar.error(f"Error al cargar dependencias: {e}")
    st.stop()

# Obtener categorías válidas de Felder
columnas_felder = [col for col in columnas_one_hot if col.startswith('Felder_')]
opciones_felder = [col.replace('Felder_', '') for col in columnas_felder]
if not opciones_felder:
    opciones_felder = ['sensorial', 'activo', 'visual', 'equilibrio', 'secuencial', 'reflexivo', 'verbal', 'intuitivo']

# Pestañas de Navegación
tab1, tab2 = st.tabs(["Predicción Individual", "Predicción por Archivo (Excel)"])

with tab1:
    st.header("Datos del Estudiante")
    felder_input = st.selectbox("Selecciona el estilo de aprendizaje (Felder):", opciones_felder, key="felder_ind")
    examen_input = st.number_input("Nota Examen de Admisión:", min_value=0.0, max_value=5.0, value=3.83, step=0.01, key="examen_ind")
    
    if st.button("Realizar Predicción", key="btn_ind"):
        try:
            df_input = pd.DataFrame({'Felder': [felder_input], 'Examen_admisión': [examen_input]})
            
            # Crear estructura One-Hot
            for col in columnas_one_hot:
                if col.startswith('Felder_'):
                    cat = col.replace('Felder_', '')
                    df_input[col] = 1.0 if felder_input == cat else 0.0
            
            df_input['Examen_admision_scaled'] = scaler.transform(df_input[['Examen_admisión']])[0][0]
            df_procesado = df_input[columnas_one_hot]
            
            pred = modelo_bagging.predict(df_procesado)[0]
            
            st.metric(label="Nota Final Estimada", value=f"{pred:.4f}")
            st.subheader("Datos Procesados para el Modelo:")
            st.dataframe(df_procesado)
            
        except Exception as e:
            st.error(f"Error al procesar la predicción: {e}")

with tab2:
    st.header("Predicción Masiva desde Excel")
    st.write("Sube un archivo Excel (.xlsx) que contenga las columnas `Felder` y `Examen_admisión`.")
    
    uploaded_file = st.file_uploader("Selecciona un archivo Excel", type=["xlsx"])
    
    if uploaded_file is not None:
        try:
            df_excel = pd.read_excel(uploaded_file)
            st.write("Vista previa de los datos cargados:")
            st.dataframe(df_excel.head())
            
            if 'Felder' in df_excel.columns and 'Examen_admisión' in df_excel.columns:
                if st.button("Procesar y Predecir Archivo"):
                    df_res = df_excel.copy()
                    df_temp = pd.DataFrame(0.0, index=df_excel.index, columns=columnas_one_hot)
                    
                    # Codificación Felder
                    for col in columnas_one_hot:
                        if col.startswith('Felder_'):
                            cat = col.replace('Felder_', '')
                            df_temp[col] = (df_excel['Felder'].astype(str).str.strip() == cat).astype(float)
                    
                    # Escalar Examen de admisión
                    df_temp['Examen_admision_scaled'] = scaler.transform(df_excel[['Examen_admisión']])
                    
                    # Predicción
                    predicciones = modelo_bagging.predict(df_temp)
                    df_res['Nota_final_estimada'] = predicciones
                    
                    st.success("¡Predicciones masivas completadas!")
                    st.dataframe(df_res)
                    
                    csv = df_res.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Descargar Resultados como CSV",
                        data=csv,
                        file_name="predicciones_curso.csv",
                        mime="text/csv"
                    )
            else:
                st.error("El archivo Excel debe contener obligatoriamente las columnas 'Felder' y 'Examen_admisión'.")
        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")
