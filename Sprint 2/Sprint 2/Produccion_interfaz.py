import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ==================================================
# CONFIGURACIÓN DE LA PÁGINA
# ==================================================
st.set_page_config(
    page_title="Dashboard de Análisis de Café",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# ESTILOS CSS
# ==================================================
st.markdown("""
<style>
.main {
    padding: 0rem 1rem;
}

/* Caja de la métrica */
div[data-testid="stMetric"] {
    background-color: #f0f2f6;
    padding: 16px;
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.08);
}

/* Título de la métrica */
div[data-testid="stMetric"] label {
    color: #2c3e50 !important;
    font-weight: 600;
}

/* Valor de la métrica */
div[data-testid="stMetric"] div {
    color: #000000 !important;
    font-size: 26px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# CARGA Y LIMPIEZA DE DATOS
# ==================================================
@st.cache_data
def load_data():
    df = pd.read_csv("reportes_cafe_clean.csv")

    # Fechas
    df["fecha_reporte"] = pd.to_datetime(df["fecha_reporte"], format="%m/%d/%Y")
    df["Fecha_clima"] = pd.to_datetime(df["Fecha_clima"], format="%m/%d/%Y")

    # Copia limpia
    df_clean = df.copy()

    # Reemplazos de valores anómalos
    df_clean["PH"] = df_clean["PH"].replace(99.9, np.nan)
    df_clean["Temperatura"] = df_clean["Temperatura"].replace(-5, np.nan)
    df_clean["Tipo_cafe"] = df_clean["Tipo_cafe"].replace("Desconocido", np.nan)

    # Conversión numérica segura
    cols_numericas = ["PH", "Temperatura", "Humedad", "hectarias"]
    for col in cols_numericas:
        if col in df_clean.columns:
            df_clean[col] = (
                df_clean[col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.replace(",", ".", regex=False)
            )
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    return df, df_clean


df, df_clean = load_data()

# ==================================================
# TÍTULO
# ==================================================
st.title("Dashboard de Análisis de Reportes de Café")
st.markdown("---")

# ==================================================
# SIDEBAR - FILTROS
# ==================================================
st.sidebar.header("Filtros")

fecha_min = df_clean["fecha_reporte"].min().date()
fecha_max = df_clean["fecha_reporte"].max().date()

fecha_rango = st.sidebar.date_input(
    "Rango de Fechas",
    value=(fecha_min, fecha_max),
    min_value=fecha_min,
    max_value=fecha_max
)

tipos_cafe = ["Todos"] + sorted(df_clean["Tipo_cafe"].dropna().unique())
tipo_sel = st.sidebar.selectbox("Tipo de Café", tipos_cafe)

ubicaciones = ["Todas"] + sorted(df_clean["ubicacion"].dropna().unique())
ubicacion_sel = st.sidebar.selectbox("Ubicación", ubicaciones)

# ==================================================
# APLICAR FILTROS
# ==================================================
df_filtrado = df_clean.copy()

if len(fecha_rango) == 2:
    df_filtrado = df_filtrado[
        (df_filtrado["fecha_reporte"].dt.date >= fecha_rango[0]) &
        (df_filtrado["fecha_reporte"].dt.date <= fecha_rango[1])
    ]

if tipo_sel != "Todos":
    df_filtrado = df_filtrado[df_filtrado["Tipo_cafe"] == tipo_sel]

if ubicacion_sel != "Todas":
    df_filtrado = df_filtrado[df_filtrado["ubicacion"] == ubicacion_sel]

# ==================================================
# MÉTRICAS PRINCIPALES
# ==================================================
st.header("Métricas Principales")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Total Reportes", f"{len(df_filtrado):,}")

ph_mean = df_filtrado["PH"].mean()
c2.metric("PH Promedio", f"{ph_mean:.2f}" if not np.isnan(ph_mean) else "N/A")

temp_mean = df_filtrado["Temperatura"].mean()
c3.metric("Temp. Prom.", f"{temp_mean:.1f} °C" if not np.isnan(temp_mean) else "N/A")

hum_mean = df_filtrado["Humedad"].mean()
c4.metric("Humedad Prom.", f"{hum_mean:.1f} %" if not np.isnan(hum_mean) else "N/A")

hec_sum = df_filtrado["hectarias"].sum()
c5.metric("Hectáreas Total", f"{hec_sum:,.0f}" if not np.isnan(hec_sum) else "N/A")

st.markdown("---")

# ==================================================
# PESTAÑAS
# ==================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Distribuciones",
    "Análisis Geográfico",
    "Análisis Temporal",
    "Análisis por Tipo",
    "Datos Crudos"
])

# ==================================================
# TAB 1 - DISTRIBUCIONES
# ==================================================
with tab1:
    st.subheader("Distribución de Tipos de Café")

    cafe_counts = df_filtrado["Tipo_cafe"].value_counts()

    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(
            values=cafe_counts.values,
            names=cafe_counts.index,
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            x=cafe_counts.index,
            y=cafe_counts.values,
            labels={"x": "Tipo de Café", "y": "Cantidad"}
        )
        st.plotly_chart(fig, use_container_width=True)

# ==================================================
# TAB 2 - GEOGRÁFICO
# ==================================================
with tab2:
    st.subheader("Reportes por Ubicación")

    ubicacion_counts = df_filtrado["ubicacion"].value_counts()

    fig = px.bar(
        x=ubicacion_counts.index,
        y=ubicacion_counts.values,
        labels={"x": "Ubicación", "y": "Cantidad"}
    )
    st.plotly_chart(fig, use_container_width=True)

    metricas_ubicacion = df_filtrado.groupby("ubicacion").agg({
        "PH": "mean",
        "Temperatura": "mean",
        "Humedad": "mean",
        "hectarias": "sum"
    }).round(2)

    st.dataframe(metricas_ubicacion, use_container_width=True)

# ==================================================
# TAB 3 - TEMPORAL
# ==================================================
with tab3:
    st.subheader("Reportes a lo largo del tiempo")

    reportes_fecha = (
        df_filtrado
        .groupby(df_filtrado["fecha_reporte"].dt.date)
        .size()
        .reset_index(name="Cantidad")
    )

    fig = px.line(
        reportes_fecha,
        x="fecha_reporte",
        y="Cantidad",
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# TAB 4 - POR TIPO
# ==================================================
with tab4:
    st.subheader("Comparación por Tipo de Café")

    for col in ["PH", "Temperatura", "Humedad", "hectarias"]:
        fig = px.box(
            df_filtrado,
            x="Tipo_cafe",
            y=col,
            color="Tipo_cafe"
        )
        st.plotly_chart(fig, use_container_width=True)

# ==================================================
# TAB 5 - DATOS CRUDOS
# ==================================================
with tab5:
    st.subheader(f"Registros mostrados: {len(df_filtrado)}")

    csv = df_filtrado.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Descargar CSV",
        csv,
        f"reportes_filtrados_{datetime.now().strftime('%Y%m%d')}.csv",
        "text/csv"
    )

    st.dataframe(df_filtrado, use_container_width=True)

