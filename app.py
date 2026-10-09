import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.impute import KNNImputer

# 1. Configuración de la página
st.set_page_config(
    page_title="Predicción de Ventas - Stacking",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Aplicación de Predicción de Ventas Semanales (Stacking Optimizado)")
st.markdown("""
Esta aplicación permite subir un archivo de datos en formato **Excel (.xlsx o .xls)** para predecir las ventas semanales.
Realiza el preprocesamiento de datos completo:
- Limpieza de outliers mediante IQR.
- Imputación de valores nulos utilizando KNN Imputer.
- Escalado Min-Max de las características predictoras utilizando el escalador entrenado.
- Predicciones con el modelo de ensamble **Stacking Regressor**.
""")

# Obtener rutas independientes del entorno de ejecución de Colab
dir_actual = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()

# Buscar el escalador y el modelo en el directorio actual o en la subcarpeta 'content'
scaler_candidates = [
    os.path.join(dir_actual, 'min_max_scaler.joblib'),
    os.path.join(dir_actual, '/Datos', 'min_max_scaler.joblib'),
    os.path.join(dir_actual, '..', 'min_max_scaler.joblib')
]
model_candidates = [
    os.path.join(dir_actual, 'Stacking_optimizado.joblib'),
    os.path.join(dir_actual, '/Datos', 'Stacking_optimizado.joblib'),
    os.path.join(dir_actual, '..', 'Stacking_optimizado.joblib')
]

scaler_path = next((p for p in scaler_candidates if os.path.exists(p)), None)
model_path = next((p for p in model_candidates if os.path.exists(p)), None)

if not scaler_path or not model_path:
    st.error("⚠️ No se encontraron los archivos del modelo o del escalador en las rutas relativas de la aplicación.")
    st.info("Por favor, asegúrate de subir 'min_max_scaler.joblib' y 'Stacking_optimizado.joblib' junto a este archivo en tu repositorio de GitHub.")
else:
    # Cargar escalador y modelo de forma segura
    scaler = joblib.load(scaler_path)
    stacking_model = joblib.load(model_path)

    # Widget para cargar archivo de datos
    archivo_subido = st.file_uploader("Sube tu archivo Excel con datos de ventas (.xlsx, .xls)", type=["xlsx", "xls"])

    if archivo_subido is not None:
        try:
            # Leer datos de Excel
            df_original = pd.read_excel(archivo_subido)
            st.success("¡Archivo cargado correctamente!")

            st.subheader("Vista previa de los datos subidos")
            st.dataframe(df_original.head(10))

            with st.spinner("Procesando características y ejecutando el Stacking optimizado..."):
                df_proc = df_original.copy()

                # Convertir objetos a categorías
                for col in df_proc.select_dtypes(include=['object']).columns:
                    df_proc[col] = df_proc[col].astype('category')

                # Limpieza de fechas si están presentes
                if 'Date' in df_proc.columns:
                    df_proc['Date'] = pd.to_datetime(df_proc['Date'], errors='coerce', format='mixed')
                    df_proc = df_proc.drop(columns=['Date'])

                # Eliminar variables demográficas e irrelevantes
                columnas_a_eliminar = ['RecordID', 'FullName', 'Phone', 'ZodiacSign', 'FavoriteColor', 'Hobby']
                df_proc.drop(columns=[col for col in columnas_a_eliminar if col in df_proc.columns], inplace=True, errors='ignore')

                # Calcular Store_Total_Sales si no está presente
                if 'Store_Total_Sales' not in df_proc.columns:
                    if 'StoreID' in df_proc.columns and 'Weekly_Sales' in df_proc.columns:
                        df_proc['Store_Total_Sales'] = df_proc.groupby('StoreID')['Weekly_Sales'].transform('sum')
                    else:
                        # Mediana histórica de entrenamiento como valor por defecto
                        df_proc['Store_Total_Sales'] = 1460766.60

                # Guardar objetivo si está para evitar outliers/imputaciones directas sobre él
                target_col = 'Weekly_Sales'
                variables_numericas = df_proc.select_dtypes(include=['number']).columns.tolist()
                if target_col in variables_numericas:
                    variables_numericas.remove(target_col)

                # Tratamiento de Outliers mediante IQR
                for col in variables_numericas:
                    Q1 = df_proc[col].quantile(0.25)
                    Q3 = df_proc[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df_proc.loc[(df_proc[col] < lower_bound) | (df_proc[col] > upper_bound), col] = np.nan

                # Imputación de nulos utilizando KNN Imputer
                cols_con_nulos = df_proc[variables_numericas].columns[df_proc[variables_numericas].isnull().any()].tolist()
                if cols_con_nulos:
                    imputer = KNNImputer(n_neighbors=5)
                    df_proc[cols_con_nulos] = imputer.fit_transform(df_proc[cols_con_nulos])

                # Rellenar cualquier nulo remanente con la mediana
                df_proc[variables_numericas] = df_proc[variables_numericas].fillna(df_proc[variables_numericas].median())

                # Reconstrucción de características que espera el scaler / modelo
                features_esperadas = [
                    'StoreID', 'DayOfWeek', 'Promotion', 'Holiday', 'Temperature',
                    'FuelPrice', 'CPI', 'Unemployment', 'Sales_3Months', 'Store_Total_Sales'
                ]

                for col_esp in features_esperadas:
                    if col_esp not in df_proc.columns:
                        df_proc[col_esp] = 0.0

                # Reordenar al formato exacto
                df_para_escalado = df_proc[features_esperadas].copy()

                # Aplicar escalado guardado
                scaled_data = scaler.transform(df_para_escalado)
                df_escalado = pd.DataFrame(scaled_data, columns=features_esperadas)

                # Predecir con el modelo final
                predicciones = stacking_model.predict(df_escalado)

                # Adjuntar predicción al original para descarga
                df_resultado = df_original.copy()
                df_resultado['Predicted_Weekly_Sales'] = predicciones

            # Mostrar Resultados
            st.subheader("✨ Resultados de las Predicciones")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Total de registros procesados", value=f"{len(df_resultado)}")
            with col2:
                st.metric(label="Promedio de Ventas Semanales Predichas", value=f"${predicciones.mean():,.2f}")

            st.dataframe(df_resultado)

            # Convertir dataframe a Excel para la descarga en memoria
            import io
            output_stream = io.BytesIO()
            with pd.ExcelWriter(output_stream, engine='xlsxwriter') as writer:
                df_resultado.to_excel(writer, index=False, sheet_name='Predicciones')

            st.download_button(
                label="📥 Descargar resultados en Excel",
                data=output_stream.getvalue(),
                file_name="predicciones_ventas_semanales.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        except Exception as e:
            st.error(f"❌ Ocurrió un error al procesar el archivo Excel: {e}")
