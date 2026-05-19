import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# ✅ LOAD DATA (robusto)
def load_data(relative_path):
    import os
    import pandas as pd

    # root progetto
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    full_path = os.path.join(base_dir, relative_path)

    print(f"📂 Loading: {full_path}")

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"❌ File NON trovato: {full_path}")

    # ✅ parsing robusto
    df = pd.read_csv(
        full_path,
        sep=r"\s+",
        engine="python",
        header=None
    )

    # ✅ DEBUG (guarda qui!)
    print("✅ Shape:", df.shape)

    # 🔴 CONTROLLO CRITICO
    if df.shape[1] != 26:
        print("❌ ERRORE: il file non è stato separato correttamente")
        print("👉 Probabile problema di formato del file")
        
        # preview contenuto reale
        with open(full_path, "r") as f:
            print("👉 Anteprima file:")
            for _ in range(3):
                print(f.readline())

        raise ValueError("❌ Parsing fallito: il file non ha 26 colonne")

    # ✅ solo se ok assegno colonne
    cols = ["unit", "cycle"] + \
           [f"op_{i}" for i in range(1, 4)] + \
           [f"sensor_{i}" for i in range(1, 22)]

    df.columns = cols

    return df


# ✅ PREPROCESSING
def add_rul(df):
    df["RUL"] = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
    return df


# ✅ FEATURE ENGINEERING
def add_features(df):
    df["sensor_5_mean"] = df["sensor_5"].rolling(5).mean()
    return df.dropna()


# ✅ TARGET
def create_target(df):
    df["label"] = (df["RUL"] <= 30).astype(int)
    return df


# ✅ TRAIN MODEL
def train_model(df):
    X = df.drop(["unit", "cycle", "RUL", "label"], axis=1)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    print("✅ Model trained")

    return model
