# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 18:41:09 2025

@author: jperezr
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la página
st.set_page_config(page_title="Clustering de Trabajadores", page_icon="📊", layout="wide")
st.title("📊 Segmentación de Trabajadores con Machine Learning")

# Carga de datos
uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

# Sección de ayuda en el sidebar
with st.sidebar:
    st.header("Configuración del Clustering")
    
    # Sección de ayuda después de los controles
    st.markdown("---")
    st.subheader("ℹ️ Ayuda")
    st.markdown("""
    **Esta aplicación permite:**
    - Segmentar trabajadores en clusters usando K-Means
    - Visualizar los grupos en 2D usando PCA
    - Analizar características de cada cluster
    - Descargar los resultados
    
    **Instrucciones:**
    1. Sube tu archivo CSV
    2. Selecciona variables numéricas
    3. Ajusta el número de clusters
    4. Explora los resultados
    
    **Variables recomendadas:**
    - Edad trabajador
    - Salario base
    - Ahorros acumulados
    
    Desarrollado por: Javier Horacio Pérez Ricárdez
    """)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Sidebar: Selección de variables y parámetros
    with st.sidebar:
        # Selección de variables
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = ["Siefore cuenta", "Siefore inversion"]
        
        numeric_cols_for_scaling = [col for col in numeric_cols if col not in categorical_cols]
        
        default_vars = ["Edad trabajador", "Bimestres cotizacion", "Salario base cotizacion", 
                       "Saldo acumulado RCV(al corte)", "Saldo acumulado ahorro voluntario(al corte)"]
        selected_vars = st.multiselect(
            "Selecciona variables numéricas para clustering",
            options=numeric_cols_for_scaling,
            default=default_vars
        )
        
        selected_categorical = st.multiselect(
            "Selecciona variables categóricas (opcional)",
            options=categorical_cols,
            default=[]
        )
        
        n_clusters = st.slider("Número de clusters", 2, 10, 3)
    
    if selected_vars and len(selected_vars) >= 2:
        # Preprocesamiento
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), selected_vars),
                ('cat', OneHotEncoder(drop='first'), selected_categorical)
            ])
        
        # Filtrar filas sin NaN
        X = df[selected_vars + selected_categorical].dropna()
        X_processed = preprocessor.fit_transform(X)
        
        # K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_processed)
        
        # --- Fusionar resultados con DataFrame original ---
        df["Cluster"] = np.nan
        df["PCA1"] = np.nan
        df["PCA2"] = np.nan
        df.loc[X.index, "Cluster"] = clusters
        
        # PCA para visualización
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed)
        df.loc[X.index, "PCA1"] = X_pca[:, 0]
        df.loc[X.index, "PCA2"] = X_pca[:, 1]
        
        # --- Visualización ---
        st.subheader("Visualización de Clusters (PCA)")
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = sns.scatterplot(
            data=df.dropna(subset=["Cluster"]),
            x="PCA1",
            y="PCA2",
            hue="Cluster",
            palette="viridis",
            alpha=0.7,
            ax=ax
        )
        plt.title(f"Clusters de Trabajadores (K={n_clusters})")
        st.pyplot(fig)
        
        # --- Estadísticas numéricas ---
        st.subheader("📌 Estadísticas por Cluster (Variables Numéricas)")
        clustered_data = df.dropna(subset=["Cluster"])
        st.dataframe(
            clustered_data.groupby("Cluster")[selected_vars].mean()
            .style.background_gradient(cmap="Blues"),
            use_container_width=True
        )
        
        # --- Distribución de Categorías ---
        if selected_categorical:
            st.subheader("📊 Distribución de Categorías por Cluster")
            for cat_var in selected_categorical:
                st.write(f"**Variable:** {cat_var}")
                freq_table = pd.crosstab(
                    index=clustered_data["Cluster"], 
                    columns=clustered_data[cat_var],
                    margins=True
                )
                st.dataframe(
                    freq_table.style.background_gradient(cmap="Greens"),
                    use_container_width=True
                )
        
        # --- Interpretación de Clusters ---
        st.subheader("🧠 Interpretación de Clusters")
        
        cluster_names = {
            0: "Perfil Conservador",
            1: "Perfil Moderado",
            2: "Perfil Agresivo"
        }
        
        for cluster_num in range(n_clusters):
            cluster_data = clustered_data[clustered_data["Cluster"] == cluster_num]
            
            st.markdown(f"### 🔹 {cluster_names.get(cluster_num, f'Cluster {cluster_num}')}")
            
            edad_mediana = cluster_data["Edad trabajador"].median()
            st.write(f"- **Edad promedio**: {edad_mediana:.0f} años")
            
            salario_mediano = cluster_data["Salario base cotizacion"].median()
            st.write(f"- **Salario mediano**: ${salario_mediano:,.0f} MXN")
            
            saldo_rcv = cluster_data["Saldo acumulado RCV(al corte)"].median()
            st.write(f"- **Ahorro para retiro (RCV)**: ${saldo_rcv:,.0f} MXN")
            
            if selected_categorical:
                siefore_counts = cluster_data["Siefore cuenta"].value_counts(normalize=True) * 100
                top_siefore = siefore_counts.idxmax()
                st.write(f"- **Siefore predominante**: {top_siefore} ({siefore_counts.max():.1f}%)")
            
            st.write("---")
        
        # --- Mostrar TODOS los datos ---
        st.subheader("🔍 Todos los datos con clusters asignados")
        st.dataframe(df, height=500)
        
        # --- Descargar resultados ---
        st.sidebar.download_button(
            label="Descargar datos completos con clusters",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="trabajadores_con_clusters_completo.csv",
            mime="text/csv"
        )
        
    else:
        st.warning("⚠️ Selecciona al menos 2 variables numéricas.")
else:
    st.info("👋 Sube un archivo CSV para comenzar.")
