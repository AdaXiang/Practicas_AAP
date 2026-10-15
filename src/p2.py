import pandas as pd

"""
Balanceo del split train del dataset viviendas_sevila.csv
"""
CSV_PATH = "./datasets/viviendas_sevilla.csv"

def analisis_balanceo(csv: str):
    df = pd.read_csv(csv)
    print(df["tipo"].value_counts())
    print(df["tipo"].value_counts(normalize=True).mul(100).round(2).astype(str) + " %")


# TODO: Undersampling: Se eliminan aleatoriamente muestras de la clase mayoritaria para igualar el número de muestras de la clase minoritaria.

# TODO: Oversampling: duplicar.

# TODO: Oversampling: generar sintéticamente nuevas muestras de la clase minoritaria mediante técnicas como SMOTE (Synthetic Minority Over-sampling Technique).

# TODO: Cost sensitive learning: asignar un mayor peso a las muestras de la clase minoritaria durante el entrenamiento del modelo para que tenga un mayor impacto en la función de pérdida.

def main():
    analisis_balanceo(CSV_PATH)

    return None

if __name__ == "__main__":
    main()