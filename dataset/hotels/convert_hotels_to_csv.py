import numpy as np
import pandas as pd
import math

# Liste der .npy-Dateien und zugehöriger Hotelwerte
npy_files = ["F1_Equal.npy", "F2_Equal.npy", "F3_Equal.npy", "F4_Equal.npy"]
hotel_values = [0, 1, 2, 3]

# Which column name is the one too much?
column_names = [
    "evaporator_inlet_water_temperature",
    "evaporator_outlet_water_temperature",
    "condenser_inlet_water_temperature",
    "condenser_outlet_water_temperature",
    "evaporator_cooling_capacity",
    "compressor_inlet_air_temperature",
    "compressor_outlet_air_temperature",
    "evaporator_inlet_air_pressure",
    "condenser_outlet_air_pressure",
    "exhaust_air_overheat_temperature",
    "main_circuit_coolant_level",
    "main_coolant_pipe_valve_opening_size",
    "compressor_load",
    "compressor_current",
    "compressor_rotational_speed",
    "compressor_voltage",
    "compressor_power",
    "compressor_inverter_temperature",
    "fault"
]

feature_names = [
    "f1",
    "f2",
    "f3",
    "f4",
    "f5",
    "f6",
    "f7",
    "f8",
    "f9",
    "f10",
    "f11",
    "f12",
    "f13",
    "f14",
    "f15",
    "f16",
    "f17",
    "fault"
]

# Daten kombinieren
data_frames = []
for file, hotel in zip(npy_files, hotel_values):
    # Lade die .npy-Datei
    data = np.load(f"./raw/{file}")

    # Erstelle einen DataFrame mit den Daten und füge die Spalte 'hotel' hinzu
    df = pd.DataFrame(data, columns=feature_names)
    df['hotel'] = hotel
    df['fault'] = df['fault'].astype(int)

    data_frames.append(df)

# Kombiniere alle DataFrames
final_df = pd.concat(data_frames, ignore_index=True)

final_df.to_csv("hotels.csv", index=False)

positive_df = final_df[final_df['fault'] == 1]
negative_df = final_df[final_df['fault'] == 0]

# Funktion zum Erzeugen von Datensätzen mit verschiedenen positiven/negativen Verhältnissen
def create_datasets(positive_df, negative_df, ratios):
    datasets = {}

    for ratio in ratios:
        # Berechne die Anzahl der negativen Labels, die benötigt werden
        num_positives_needed = int(len(negative_df) * 1/ratio)

        # Zufällig eine Teilmenge der negativen Labels auswählen
        sampled_positives = negative_df.sample(num_positives_needed, random_state=42)

        # Kombiniere die positiven und negativen Datensätze
        dataset = pd.concat([negative_df, sampled_positives], ignore_index=True)

        # Mische die Reihenfolge der Daten
        dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

        # Speichern des Datensatzes
        datasets[ratio] = dataset

    return datasets


# Verhältnisse definieren
ratios = [4, 10, 20, 30]

# Datensätze mit den gewünschten Verhältnissen erstellen
datasets = create_datasets(positive_df, negative_df, ratios)

# Speichern der Datensätze als separate CSV-Dateien
for ratio, dataset in datasets.items():
    dataset.to_csv(f"hotels_ratio_1-{ratio}.csv", index=False)
