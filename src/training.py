import argparse
import os
import pandas as pd
import joblib

from function import (
    load_data,
    add_rul,
    add_features,
    create_target,
    train_model
)

DATA_DIR = "data/raw"
MODEL_PATH = "models/model.pkl"


def process_single_file(file_name):
    """Processa un singolo file"""
    path = os.path.join(DATA_DIR, file_name)

    print(f"📂 Processing: {path}")

    df = load_data(path)
    df = add_rul(df)

    return df


def process_all_files():
    """Unisce tutti i file train"""
    dfs = []

    for file in os.listdir(DATA_DIR):
        if file.startswith("train"):
            df = process_single_file(file)
            dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="File singolo")
    parser.add_argument("--all", action="store_true")

    args = parser.parse_args()

    # scelta input
    if args.all:
        df = process_all_files()
    elif args.file:
        df = process_single_file(args.file)
    else:
        raise ValueError("❌ Usa --file oppure --all")

    # feature + target
    df = add_features(df)
    df = create_target(df)

    # training
    model = train_model(df)

    # salva modello
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"✅ Modello salvato in {MODEL_PATH}")


if __name__ == "__main__":
    main()
