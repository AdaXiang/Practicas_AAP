import os
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import imblearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder

# Rutas de datos y caché
RAW_CSV_PATH = "./datasets/viviendas_sevilla.csv"
RAW_CACHE_PATH = "./datasets/cache_viviendas_raw.parquet"
CLEAN_CACHE_PATH = "./datasets/cache_viviendas_clean.parquet"


def get_raw_data(force_reload: bool = False) -> pd.DataFrame:
    """Carga el dataset sucio desde la caché si existe; si no, lee el CSV y lo cachea."""
    if not force_reload and os.path.exists(RAW_CACHE_PATH):
        return pd.read_parquet(RAW_CACHE_PATH)

    df = pd.read_csv(RAW_CSV_PATH)
    df.to_parquet(RAW_CACHE_PATH, index=False)
    return df

def guardar_dataframe_csv(
    df: pd.DataFrame,
    nombre_archivo: str = "viviendas_sevilla.csv",
    carpeta_destino: str = "viviendas_sevilla_csv",
) -> str:
    # 1. Crear la carpeta si no existe
    os.makedirs(carpeta_destino, exist_ok=True)

    # 2. Asegurar la extensión .csv
    if not nombre_archivo.endswith(".csv"):
        nombre_archivo += ".csv"

    # 3. Construir la ruta y exportar sin el índice
    ruta_completa = os.path.join(carpeta_destino, nombre_archivo)
    df.to_csv(ruta_completa, index=False)

    print(f"Archivo guardado correctamente en: {ruta_completa}")
    return ruta_completa

def generar_graficas(df, nombre_dataset="viviendas_sevilla"):
    # 1. Crear la carpeta si no existe (ej: "viviendas_sevilla_graficas")
    carpeta_destino = f"{nombre_dataset}_graficas"
    os.makedirs(carpeta_destino, exist_ok=True)

    # --- 1. HISTOGRAMAS (Distribución) ---
    print("\n--- Generando Histogramas ---")
    numeric_cols = df.select_dtypes(include="number").columns
    n_cols = len(numeric_cols)

    fig, axes = plt.subplots(nrows=2, ncols=(n_cols + 1) // 2, figsize=(16, 9))
    axes = axes.flatten()

    for idx, col in enumerate(numeric_cols):
        serie = df[col].dropna()

        # Determinar si la variable es discreta
        es_discreta = bool(pd.api.types.is_integer_dtype(df[col]) or (df[col].nunique() <= 15))

        if es_discreta:
            sns.histplot(serie, ax=axes[idx], discrete=True, kde=False)
        else:
            # Comprobar dispersión y usar base 10 explícita si aplica (evita pasar np.bool_)
            tiene_outliers = bool((serie.max() / (serie.quantile(0.75) + 1e-5)) > 20)
            usar_log = 10 if (tiene_outliers and (serie > 0).all()) else False

            sns.histplot(serie, ax=axes[idx], bins=30, kde=False, log_scale=usar_log)

        axes[idx].set_title(col)

    for j in range(n_cols, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(os.path.join(carpeta_destino, "histogramas.png"), dpi=300)
    plt.show()
    plt.close()

    # --- 2. BOXPLOTS (Valores anómalos / Outliers) ---
    print("\n¿Existen valores anómalos?")
    numeric_cols = df.select_dtypes(include="number").columns
    
    if len(numeric_cols) > 0:
        n_cols = len(numeric_cols)
        # Cuadrícula de subplots 
        fig, axes = plt.subplots(nrows=2, ncols=(n_cols + 1) // 2, figsize=(15, 8))
        axes = axes.flatten()

        for idx, col in enumerate(numeric_cols):
            # Cada columna tiene su propio eje con su escala adaptada
            df.boxplot(column=col, ax=axes[idx])
            axes[idx].set_title(col, fontsize=11, fontweight="bold")
            axes[idx].grid(True, linestyle="--", alpha=0.6)

        # Ocultar ejes vacíos sobrantes si el número es impar
        for j in range(len(numeric_cols), len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle(f"Boxplots (Escalas independientes) - {nombre_dataset}", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(carpeta_destino, "boxplots.png"), dpi=300)
        plt.show()
        plt.close()

    # --- 3. MATRIZ DE CORRELACIÓN (Heatmap) ---
    print("\n¿Hay correlaciones?")
    corr = df.corr(numeric_only=True)
    print(corr.round(2))

    if not corr.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
        plt.title(f"Matriz de Correlación - {nombre_dataset}")
        plt.tight_layout()
        plt.savefig(os.path.join(carpeta_destino, "correlacion_heatmap.png"), dpi=300)
        plt.show()
        plt.close()

    print(f"\n[OK] Gráficas guardadas correctamente en la carpeta: '{carpeta_destino}'")


def eda(df: pd.DataFrame) -> None:
    print("\n--- EDA (ANÁLISIS EXPLORATORIO) ---")
    print(f"¿Cuántas filas tenemos?\n{df.shape[0]}\n")

    print("¿Qué valores existen?")
    print(df.shape[1])

    print("\n¿Qué tipos de valores tienen?")
    print(df.dtypes)

    print("\n¿Hay valores ausentes?")
    print(df.isnull().sum())
    for col in df.columns[df.isnull().any()]:
        guardar_dataframe_csv(df[df[col].isnull()], f"nulos_{col}.csv")

    print("\n¿Hay duplicados?")
    print(df.duplicated().sum())
    duplicados = df[df.duplicated(keep=False)]
    print(duplicados.sort_values(by=list(df.columns)))

    print("\n¿Hay valores únicos?")
    print(df.nunique())
    print("\n")
    # Muestra un diccionario con los valores únicos por columna
    cols_str = df.select_dtypes(include=["object", "string"]).columns
    unicos_filtrados = {
            col: df[col].unique() 
            for col in cols_str 
            if df[col].nunique() <= 15
        }
    for col, valores in unicos_filtrados.items():
        print(f"Columna: {col} ({len(valores)} únicos)")
        print(valores)
        print("\n" + "-" * 40 + "\n")

    print("\n¿Cuál es la distribución de las variables?")
    """
     df.describe(): Genera una tabla con 8 métricas estadísticas clave para cada variable numérica:
     count: Cantidad de valores no nulos.mean: Media o promedio aritmético.
     std: Desviación estándar (dispersión de los datos).
     min: Valor mínimo observado.25%: Primer cuartil ($Q_1$, el 25% de los datos queda por debajo).
     50%: Mediana o segundo cuartil ($Q_2$, el centro exacto).
     75%: Tercer cuartil ($Q_3$, el 75% de los datos queda por debajo).
     max: Valor máximo registrado.
     T (Transpose): Invierte los ejes.
    """
    print(df.describe().T)
   
    # print("\n¿Existen valores anómalos?")
    # print("\n¿Hay correlaciones?")
    generar_graficas(df)

    print("\n¿Hay variables muy relacionadas entre sí?")
    print("\n Los baños y las habitaciones = 0.81")
    

def limpieza(df: pd.DataFrame, force_reload: bool = False) -> pd.DataFrame:
    if not force_reload and os.path.exists(CLEAN_CACHE_PATH):
       #print("Dataset limpio recuperado desde caché.")
        return pd.read_parquet(CLEAN_CACHE_PATH)

    #print("Ejecutando pipeline de limpieza de datos...")
    clean_df = df.copy()

    # 1. Eliminar duplicados
    clean_df = clean_df.drop_duplicates()
    
    # 2. Imputar valores nulos (por ejemplo, con la mediana)
    imputer = SimpleImputer(strategy="median")

    # Si acensor es nulo, lo remplazamos por 0, ya que no tiene ascensor
    clean_df['ascensor'] = clean_df['ascensor'].fillna("No")

    # Si superficie es nulo, lo remplazamos por la mediana de superficie
    clean_df['superficie_m2'] = imputer.fit_transform(clean_df[['superficie_m2']])

    # Si el año de construcción es nulo, lo remplazamos utilizando el knn
    cols_knn = [
    'superficie_m2',
    'habitaciones',
    'banos',
    'planta',
    'ano_construccion',
    'precio',
]

    # 1. Escalamos las variables al rango [0, 1] para equilibrar distancias
    scaler = MinMaxScaler()
    datos_escalados = scaler.fit_transform(clean_df[cols_knn])

    # 2. Imputamos con KNN
    knn_imputer = KNNImputer(n_neighbors=5)
    datos_imputados_escalados = knn_imputer.fit_transform(datos_escalados)

    # 3. Desescalamos a los valores reales originales
    datos_finales = scaler.inverse_transform(datos_imputados_escalados)

    # 4. Extraemos ano_construccion, redondeamos y convertimos a entero
    idx_ano = cols_knn.index('ano_construccion')
    clean_df['ano_construccion'] = np.round(datos_finales[:, idx_ano]).astype(int)

    # Modificar o eliminar ouliers
    # Eliminar superficie 9999 
    clean_df = clean_df[clean_df['superficie_m2'] != 9999]

    # Modificar estudio de 25 habitaciones a 0 
    clean_df.loc[clean_df['habitaciones'] == 25, 'habitaciones'] = 0

    # Eliminar precios 2M MACARENA CHALET y 1.2 CERRO MOMENTO CHALET (ojo con el precio en TRIANA 2M)
    clean_df = clean_df[~((clean_df['precio'] == 2000000) & (clean_df['barrio'] == 'Macarena'))]
    clean_df = clean_df[~((clean_df['precio'] == 1200000) & (clean_df['barrio'] == 'Cerro Amate'))]

    # Guardar en caché el dataset ya saneado
    clean_df.to_parquet(CLEAN_CACHE_PATH, index=False)
    print(f"Dataset limpio guardado en: {CLEAN_CACHE_PATH}")
    return clean_df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # precio_m2 = precio / superficie_m2
    df['precio_m2'] = df['precio'] / df['superficie_m2']

    # precio_habitacion = precio / habitaciones (evitar división por cero)
    df['precio_habitacion'] = df.apply(lambda row: row['precio'] / row['habitaciones'] if row['habitaciones'] > 0 else np.nan, axis=1)

    # dividr la fehca en año, mes y día
    df['fecha_venta'] = pd.to_datetime(df['fecha_venta'], format='%d/%m/%Y', errors='coerce')
    df['año'] = df['fecha_venta'].dt.year
    df['mes'] = df['fecha_venta'].dt.month
    df['dia'] = df['fecha_venta'].dt.day
    df['dia_semana'] = df['fecha_venta'].dt.dayofweek  # 0 = Lunes, 6 = Domingo
    df['es_fin_de_semana'] = df['fecha_venta'].dt.dayofweek.isin([5, 6]).astype(int)
    df['trimestre'] = df['fecha_venta'].dt.quarter

    return df

def codificacion(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # 1. Codificar variables nominales con OneHotEncoder
    columnas_cat = ['barrio', 'tipo', 'estado']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    
    encoded_matrix = encoder.fit_transform(df[columnas_cat])
    encoded_cols = encoder.get_feature_names_out(columnas_cat)
    encoded_df = pd.DataFrame(encoded_matrix, columns=encoded_cols, index=df.index)
    
    # Unir las nuevas columnas y eliminar las originales
    df = pd.concat([df.drop(columns=columnas_cat), encoded_df], axis=1)

    # Codificar la variable 'ascensor' a 0 y 1
    df['ascensor'] = (
    df['ascensor']
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace('í', 'i')
    .map({'no': 0, 'si': 1})
    )

    return df

def transformacion(df: pd.DataFrame) -> pd.DataFrame:
    # Aplicar transformaciones logarítmicas a las variables numéricas
    df['precio_log'] = np.log1p(df['precio'])
    df['superficie_log'] = np.log1p(df['superficie_m2'])

    return df

def estandarizacion(df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    df[['superficie_log_std']] = scaler.fit_transform(df[['superficie_log']])
    df[['precio_log_std']] = scaler.fit_transform(df[['precio_log']])

    return df

def main():
    # 1. Obtención de datos con caché
    df_raw = get_raw_data()
    print("Primeras filas del dataset:")
    print(df_raw.head())

    # 2. Exploración
    eda(df_raw)

    # 3. Limpieza y guardado en caché (pon force_reload=True si cambias las reglas de limpieza)
    df_clean = limpieza(df_raw, force_reload=False)

    # 4. Feature Engineering
    df_features = feature_engineering(df_clean)

    # 5. Codificación
    df_encoded = codificacion(df_features)

    # 6. Tranformación (logaritmo) de variables numéricas
    df_encoded = transformacion(df_encoded)

    # 7. Estandarización
    df_final = estandarizacion(df_encoded)

    # Guardar el dataset final en CSV
    guardar_dataframe_csv(df_final, nombre_archivo="viviendas_sevilla_final.csv", carpeta_destino="viviendas_sevilla_csv")


if __name__ == "__main__":
    main()