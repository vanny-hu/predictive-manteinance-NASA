import argparse
import os
import pandas as pd
import joblib

from function import load_data, add_features

DATA_DIR = "data/raw"
MODEL_PATH = "models/model.pkl"


def process_file(file_name):
    """Prepara i dati per la predizione"""
    path = os.path.join(DATA_DIR, file_name)

    print(f"📂 Processing: {path}")

    df = load_data(path)

    # solo feature (NO RUL NO label)
    df = add_features(df)

    X = df.drop(["unit", "cycle"], axis=1, errors="ignore")

    return X


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="File test")

    args = parser.parse_args()

    if not args.file:
        raise ValueError("❌ Devi specificare --file")

    # controlla modello
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("❌ Modello non trovato. Fai prima train.py")

    model = joblib.load(MODEL_PATH)

    # prepara dati
    X = process_file(args.file)

    # predizione
    predictions = model.predict(X)

    # salva output
    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/predictions.csv"

    pd.DataFrame({"prediction": predictions}).to_csv(output_path, index=False)

    print("✅ Predizioni completate")
    print(f"✅ Salvate in: {output_path}")
    print("Sample:", predictions[:10])


if __name__ == "__main__":
    main()