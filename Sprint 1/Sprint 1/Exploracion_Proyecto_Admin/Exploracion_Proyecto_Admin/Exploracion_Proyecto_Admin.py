import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde
import os
from datetime import datetime
import pathlib

# Obtener el directorio donde está el script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Crear directorio de salida
output_dir = "visualizaciones_cafe"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Verificar si el archivo existe
csv_file = 'reportes_cafe_clean.csv'
if not os.path.exists(csv_file):
    print(f"ERROR: No se encuentra el archivo '{csv_file}'")
    print(f"Directorio actual: {os.getcwd()}")
    print(f"Archivos disponibles: {os.listdir('.')}")
    exit(1)

df = pd.read_csv(csv_file)

print("=" * 60)
print("ANALISIS EXPLORATORIO DE DATOS - REPORTES DE CAFE")
print("=" * 60)

print("\nINFORMACION DEL DATASET:")
print("-" * 40)
print(f"Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")

print("\nPRIMERAS FILAS:")
print(df.head())

print("\nTIPOS DE DATOS:")
print(df.dtypes)

print("\nVALORES NULOS:")
print(df.isnull().sum())

df['fecha_reporte'] = pd.to_datetime(df['fecha_reporte'], format='%m/%d/%Y')
df['Fecha_clima'] = pd.to_datetime(df['Fecha_clima'], format='%m/%d/%Y')

print("\nValores anomalos detectados:")
print(f"PH con valor 99.9: {df[df['PH'] == 99.9].shape[0]} registros")
print(f"Temperatura con valor -5: {df[df['Temperatura'] == -5].shape[0]} registros")

df_clean = df.copy()
df_clean['PH'] = df_clean['PH'].replace(99.9, np.nan)
df_clean['Temperatura'] = df_clean['Temperatura'].replace(-5, np.nan)
df_clean['Tipo_cafe'] = df_clean['Tipo_cafe'].replace('Desconocido', np.nan)

print(f"\nRegistros despues de limpieza: {df_clean.shape[0]}")

print("\n" + "=" * 60)
print("RESUMEN ESTADISTICO")
print("=" * 60)

print("\nESTADISTICAS DESCRIPTIVAS (variables numericas):")
numeric_cols = ['PH', 'hectarias', 'Humedad', 'Temperatura']
print(df_clean[numeric_cols].describe())

print("\nESTADISTICAS DESCRIPTIVAS (variables categoricas):")
categorical_cols = ['Tipo_cafe', 'ubicacion']
for col in categorical_cols:
    print(f"\n{col}:")
    print(df_clean[col].value_counts(dropna=False).head())

print("\n" + "=" * 60)
print("CREANDO VISUALIZACIONES")
print("=" * 60)

fig_size = (12, 8)

def save_fig(fig, name):
    path = os.path.join(output_dir, name)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Guardado: {name}")
    plt.close(fig)

print("\nCreando visualizacion 1: Distribucion de tipos de cafe...")
fig, axes = plt.subplots(1, 2, figsize=fig_size)

cafe_counts = df_clean['Tipo_cafe'].value_counts()
bars = axes[0].bar(cafe_counts.index, cafe_counts.values)
axes[0].set_title('Distribucion de Tipos de Cafe', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Tipo de Cafe')
axes[0].set_ylabel('Cantidad de Reportes')
axes[0].tick_params(axis='x', rotation=45)

for bar in bars:
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width() / 2., height + 5,
                 f'{int(height)}', ha='center', va='bottom')

colors = plt.cm.Set3(np.linspace(0, 1, len(cafe_counts)))
wedges, texts, autotexts = axes[1].pie(cafe_counts.values, labels=cafe_counts.index,
                                       autopct='%1.1f%%', colors=colors,
                                       startangle=90, textprops={'fontsize': 10})
axes[1].set_title('Proporcion de Tipos de Cafe', fontsize=14, fontweight='bold')

plt.tight_layout()
save_fig(fig, '01_distribucion_tipos_cafe.png')

print("\nCreando visualizacion 2: Distribucion geografica...")
fig, ax = plt.subplots(figsize=fig_size)

ubicacion_counts = df_clean['ubicacion'].value_counts()
bars = ax.bar(ubicacion_counts.index, ubicacion_counts.values, color='sandybrown')
ax.set_title('Distribucion de Reportes por Ubicacion', fontsize=14, fontweight='bold')
ax.set_xlabel('Ubicacion')
ax.set_ylabel('Cantidad de Reportes')
ax.tick_params(axis='x', rotation=45)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., height + 3,
            f'{int(height)}', ha='center', va='bottom')

plt.tight_layout()
save_fig(fig, '02_distribucion_ubicacion.png')

print("\nCreando visualizacion 3: Distribucion de variables numericas...")
fig, axes = plt.subplots(2, 2, figsize=fig_size)
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    if col in df_clean.columns and df_clean[col].notna().sum() > 0:
        data = df_clean[col].dropna()
        ax.hist(data, bins=30, alpha=0.7, density=True, color='skyblue', edgecolor='black')

        kde = gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 1000)
        ax.plot(x_range, kde(x_range), 'r-', linewidth=2)

        mean_val = data.mean()
        median_val = data.median()
        ax.axvline(mean_val, color='green', linestyle='--', linewidth=2, label=f'Media: {mean_val:.2f}')
        ax.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Mediana: {median_val:.2f}')

        ax.set_title(f'Distribucion de {col}', fontsize=12, fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel('Densidad')
        ax.legend()
    else:
        ax.text(0.5, 0.5, f'No hay datos\npara {col}',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(f'Distribucion de {col}', fontsize=12, fontweight='bold')

plt.tight_layout()
save_fig(fig, '03_distribucion_numericas.png')

print("\nCreando visualizacion 4: Comparacion por tipo de cafe...")
fig, axes = plt.subplots(2, 2, figsize=fig_size)
axes = axes.flatten()

for i, col in enumerate(numeric_cols[:4]):
    if col in df_clean.columns:
        data_for_plot = df_clean.dropna(subset=[col, 'Tipo_cafe'])
        boxplot_data = [data_for_plot[data_for_plot['Tipo_cafe'] == tipo][col]
                        for tipo in data_for_plot['Tipo_cafe'].unique()]

        bp = axes[i].boxplot(boxplot_data, patch_artist=True)

        colors = plt.cm.Pastel1(np.linspace(0, 1, len(boxplot_data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        axes[i].set_title(f'{col} por Tipo de Cafe', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Tipo de Cafe')
        axes[i].set_ylabel(col)

        if len(data_for_plot['Tipo_cafe'].unique()) <= 6:
            axes[i].set_xticks(range(1, len(data_for_plot['Tipo_cafe'].unique()) + 1))
            axes[i].set_xticklabels(data_for_plot['Tipo_cafe'].unique(), rotation=45)
        else:
            axes[i].set_xticklabels([])
            axes[i].set_xlabel('Tipos de Cafe (demasiados para mostrar)')

plt.tight_layout()
save_fig(fig, '04_boxplots_tipos_cafe.png')

print("\nCreando visualizacion 6: Mapa de calor temporal...")
fig, ax = plt.subplots(figsize=(14, 6))

df_clean['fecha'] = df_clean['fecha_reporte'].dt.date
heatmap_data = df_clean.groupby(['fecha', 'ubicacion']).size().unstack(fill_value=0)
heatmap_data = heatmap_data.sort_index()

sns.heatmap(heatmap_data.T, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Numero de Reportes'})

ax.set_title('Actividad de Reportes por Fecha y Ubicacion', fontsize=14, fontweight='bold')
ax.set_xlabel('Fecha')
ax.set_ylabel('Ubicacion')

plt.xticks(rotation=45)
plt.tight_layout()
save_fig(fig, '05_mapa_calor_temporal.png')

print("\n" + "=" * 60)
print("RESUMEN DEL ANALISIS")
print("=" * 60)

print(f"\nVISUALIZACIONES CREADAS: 5 graficos guardados en la carpeta '{output_dir}/'")
print("\nPRINCIPALES HALLAZGOS:")

ph_mean = df_clean['PH'].mean() if 'PH' in df_clean.columns else None
temp_mean = df_clean['Temperatura'].mean() if 'Temperatura' in df_clean.columns else None
humedad_mean = df_clean['Humedad'].mean() if 'Humedad' in df_clean.columns else None

if ph_mean is not None:
    print(f"1. PH promedio: {ph_mean:.2f}")

if temp_mean is not None:
    print(f"2. Temperatura promedio: {temp_mean:.1f}C")

if humedad_mean is not None:
    print(f"3. Humedad promedio: {humedad_mean:.1f}%")

if 'Tipo_cafe' in df_clean.columns:
    cafe_principal = df_clean['Tipo_cafe'].value_counts().index[0] if not df_clean['Tipo_cafe'].isna().all() else "Ninguno"
    print(f"4. Tipo de cafe mas comun: {cafe_principal}")
else:
    cafe_principal = "N/A"

if 'ubicacion' in df_clean.columns:
    ubicacion_principal = df_clean['ubicacion'].value_counts().index[0]
    print(f"5. Ubicacion con mas reportes: {ubicacion_principal}")
else:
    ubicacion_principal = "N/A"

print("\nCALIDAD DE DATOS:")
print(f"   - Registros totales: {df.shape[0]}")
print(f"   - Valores nulos en PH: {df_clean['PH'].isna().sum()} ({df_clean['PH'].isna().sum() / df.shape[0] * 100:.1f}%)")
print(f"   - Valores nulos en Temperatura: {df_clean['Temperatura'].isna().sum()} ({df_clean['Temperatura'].isna().sum() / df.shape[0] * 100:.1f}%)")
print(f"   - Tipos de cafe desconocidos: {(df['Tipo_cafe'] == 'Desconocido').sum()}")

print("\n" + "=" * 60)
print("ANALISIS COMPLETADO CON EXITO")
print("=" * 60)
print(f"\nTodos los graficos han sido guardados en la carpeta: '{output_dir}/'")
print("Archivos creados:")
for i in range(1, 7):
    if i != 5:  # No se creó el gráfico 5
        print(f"   - 0{i}_*.png")

folder_path = pathlib.Path(output_dir).resolve()
print(f"\nRuta completa: {folder_path}")
