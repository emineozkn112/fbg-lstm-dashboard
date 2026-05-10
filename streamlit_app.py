import json
import os
from collections import Counter
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential, load_model


st.set_page_config(page_title="FBG LSTM Dashboard", layout="wide")


def get_project_dir():
    script_dir = Path(__file__).resolve().parent
    candidates = [script_dir, script_dir.parent, Path.cwd()]
    for candidate in candidates:
        if (candidate / "fbg_filtered_dataset.csv").exists():
            return candidate
    return script_dir.parent


PROJECT_DIR = get_project_dir()
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "fbg_lstm_model.keras"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
ENCODER_PATH = MODEL_DIR / "label_encoder.joblib"
CONFIG_PATH = MODEL_DIR / "model_config.json"
DEFAULT_CSV = PROJECT_DIR / "fbg_filtered_dataset.csv"


def load_dataset(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file), uploaded_file.name
    return pd.read_csv(DEFAULT_CSV), str(DEFAULT_CSV)


def make_windows(df, feature_column, label_column, window_size):
    x_values = df[[feature_column]].values
    y_values = df[label_column].values

    x_windows = []
    y_windows = []

    for i in range(len(x_values) - window_size):
        window = x_values[i : i + window_size]
        labels = y_values[i : i + window_size]
        dominant_label = Counter(labels).most_common(1)[0][0]
        x_windows.append(window)
        y_windows.append(dominant_label)

    return np.array(x_windows), np.array(y_windows)


def scale_train_test(x_train, x_test):
    scaler = StandardScaler()
    feature_count = x_train.shape[2]

    x_train_2d = x_train.reshape(-1, feature_count)
    x_test_2d = x_test.reshape(-1, feature_count)

    x_train_scaled = scaler.fit_transform(x_train_2d).reshape(x_train.shape)
    x_test_scaled = scaler.transform(x_test_2d).reshape(x_test.shape)

    return x_train_scaled, x_test_scaled, scaler


def build_lstm_model(window_size, feature_count, class_count):
    model = Sequential()
    model.add(Input(shape=(window_size, feature_count)))
    model.add(LSTM(64))
    model.add(Dropout(0.25))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(class_count, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def save_artifacts(model, scaler, encoder, config):
    MODEL_DIR.mkdir(exist_ok=True)
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    CONFIG_PATH.write_text(json.dumps(config, indent=4), encoding="utf-8")


def load_artifacts():
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return model, scaler, encoder, config


def plot_signal(df, feature_column):
    fig, ax = plt.subplots(figsize=(12, 4))
    x_axis = df["time"] if "time" in df.columns else df.index
    ax.plot(x_axis, df[feature_column], label=feature_column)
    ax.set_xlabel("Zaman")
    ax.set_ylabel("Sinyal değeri")
    ax.set_title("FBG Sinyal Grafiği")
    ax.grid(True)
    ax.legend()
    return fig


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="Eğitim Doğruluğu")
    axes[0].plot(history.history["val_accuracy"], label="Doğrulama Doğruluğu")
    axes[0].set_title("Model Doğruluk Grafiği")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Eğitim Kaybı")
    axes[1].plot(history.history["val_loss"], label="Doğrulama Kaybı")
    axes[1].set_title("Model Kayıp Grafiği")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True)
    axes[1].legend()

    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(cm, display_labels=class_names)
    display.plot(cmap="Blues", values_format="d", ax=ax)
    ax.set_title("Confusion Matrix")
    return fig


def store_model_in_session(model, scaler, encoder, config):
    st.session_state["model"] = model
    st.session_state["scaler"] = scaler
    st.session_state["encoder"] = encoder
    st.session_state["config"] = config


def model_is_ready():
    return all(
        key in st.session_state
        for key in ["model", "scaler", "encoder", "config"]
    )


def predict_window(values):
    model = st.session_state["model"]
    scaler = st.session_state["scaler"]
    encoder = st.session_state["encoder"]
    config = st.session_state["config"]

    window_size = config["window_size"]
    feature_count = config["feature_count"]

    values = np.array(values, dtype=np.float32).reshape(window_size, feature_count)
    values_2d = values.reshape(-1, feature_count)
    values_scaled = scaler.transform(values_2d).reshape(1, window_size, feature_count)

    probabilities = model.predict(values_scaled, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_label = encoder.inverse_transform([predicted_index])[0]
    confidence = float(probabilities[predicted_index])

    return predicted_label, confidence, probabilities


st.title("FBG Sensör Verisi ile LSTM Hasar Tespiti Dashboard")

st.sidebar.header("Veri ve Model")
uploaded_file = st.sidebar.file_uploader("CSV dosyası yükle", type=["csv"])

try:
    df, data_source = load_dataset(uploaded_file)
except FileNotFoundError:
    st.error("Varsayılan CSV bulunamadı. Lütfen sol menüden CSV dosyası yükle.")
    st.stop()

numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
feature_candidates = [col for col in numeric_columns if col != "time"]

if not feature_candidates:
    st.error("CSV içinde sayısal sinyal sütunu bulunamadı.")
    st.stop()

default_feature_index = 0
if "delta_lambda_filtered" in feature_candidates:
    default_feature_index = feature_candidates.index("delta_lambda_filtered")

feature_column = st.sidebar.selectbox(
    "Modelde kullanılacak sinyal sütunu",
    feature_candidates,
    index=default_feature_index,
)

label_column = "label"
has_label = label_column in df.columns

window_size = st.sidebar.slider("Pencere boyutu", 10, 80, 30, 5)
epochs = st.sidebar.slider("Epoch sayısı", 5, 50, 20, 5)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Satır Sayısı", len(df))
col2.metric("Sütun Sayısı", len(df.columns))
col3.metric("Sinyal Sütunu", feature_column)
col4.metric("Etiket Var mı?", "Evet" if has_label else "Hayır")

st.caption(f"Kullanılan veri: {data_source}")

st.subheader("Veri Ön İzleme")
preview_rows = st.slider("Gösterilecek satır sayısı", 5, min(len(df), 100), 10)
st.dataframe(df.head(preview_rows), use_container_width=True)

st.subheader("Sinyal Grafiği")
st.pyplot(plot_signal(df, feature_column))

if has_label:
    st.subheader("Etiket Dağılımı")
    label_counts = df[label_column].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(label_counts.index, label_counts.values)
    ax.set_xlabel("Etiket")
    ax.set_ylabel("Adet")
    ax.set_title("Sınıf Dağılımı")
    ax.grid(axis="y")
    st.pyplot(fig)
    st.dataframe(label_counts.rename("adet"), use_container_width=True)
else:
    st.info("Bu CSV içinde label sütunu yok. Eğitim yapılamaz ama kayıtlı modelle tahmin yapılabilir.")

st.subheader("Model Eğitimi ve Kaydetme")

train_disabled = not has_label or len(df) <= window_size

if train_disabled:
    st.warning("Eğitim için CSV içinde label sütunu olmalı ve veri pencere boyutundan uzun olmalı.")

if st.button("Yeni / Güncel Veri ile LSTM Modelini Eğit ve Kaydet", disabled=train_disabled):
    x_windows, y_text = make_windows(df, feature_column, label_column, window_size)

    encoder = LabelEncoder()
    y_numeric = encoder.fit_transform(y_text)

    x_train, x_test, y_train, y_test = train_test_split(
        x_windows,
        y_numeric,
        test_size=0.2,
        random_state=42,
        stratify=y_numeric,
    )

    x_train_scaled, x_test_scaled, scaler = scale_train_test(x_train, x_test)

    model = build_lstm_model(
        window_size=x_train_scaled.shape[1],
        feature_count=x_train_scaled.shape[2],
        class_count=len(encoder.classes_),
    )

    with st.spinner("Model eğitiliyor..."):
        history = model.fit(
            x_train_scaled,
            y_train,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            verbose=0,
        )

    test_loss, test_accuracy = model.evaluate(x_test_scaled, y_test, verbose=0)
    probabilities = model.predict(x_test_scaled, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    cm = confusion_matrix(y_test, y_pred)

    config = {
        "feature_column": feature_column,
        "label_column": label_column,
        "window_size": int(window_size),
        "feature_count": int(x_train_scaled.shape[2]),
        "class_names": encoder.classes_.tolist(),
        "source": str(data_source),
    }

    save_artifacts(model, scaler, encoder, config)
    store_model_in_session(model, scaler, encoder, config)

    st.success("Model eğitildi ve models klasörüne kaydedildi.")

    result_col1, result_col2, result_col3 = st.columns(3)
    result_col1.metric("Test Accuracy", round(float(test_accuracy), 4))
    result_col2.metric("Test Loss", round(float(test_loss), 4))
    result_col3.metric("Pencere Sayısı", len(x_windows))

    st.pyplot(plot_history(history))
    st.pyplot(plot_confusion_matrix(cm, encoder.classes_))

    report = classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

st.subheader("Kaydedilmiş Modeli Kullanma")

load_disabled = not (
    MODEL_PATH.exists()
    and SCALER_PATH.exists()
    and ENCODER_PATH.exists()
    and CONFIG_PATH.exists()
)

if st.button("Kayıtlı Modeli Yükle", disabled=load_disabled):
    model, scaler, encoder, config = load_artifacts()
    store_model_in_session(model, scaler, encoder, config)
    st.success("Kayıtlı model yüklendi.")

if load_disabled:
    st.info("Henüz kayıtlı model bulunamadı. Önce modeli eğitip kaydet.")

if model_is_ready():
    config = st.session_state["config"]
    st.write("Aktif model ayarları:")
    st.json(config)

st.subheader("Canlı Tahmin Paneli")

if not model_is_ready():
    st.warning("Canlı tahmin için önce model eğit veya kayıtlı modeli yükle.")
else:
    config = st.session_state["config"]
    active_feature = config["feature_column"]
    active_window_size = int(config["window_size"])

    if active_feature not in df.columns:
        st.error(f"Aktif model {active_feature} sütununu bekliyor, ancak bu CSV içinde yok.")
    else:
        st.write(
            f"Model son {active_window_size} ölçümü kullanarak tahmin yapacak. "
            f"Kullanılan sütun: {active_feature}"
        )

        prediction_mode = st.radio(
            "Tahmin veri kaynağı",
            ["CSV içindeki son ölçümler", "Manuel değer gir"],
            horizontal=True,
        )

        if prediction_mode == "CSV içindeki son ölçümler":
            if len(df) < active_window_size:
                st.error("CSV, modelin beklediği pencere boyutundan kısa.")
            else:
                latest_values = df[active_feature].tail(active_window_size).values
                st.line_chart(pd.DataFrame({active_feature: latest_values}))

                if st.button("Son Ölçümlerle Tahmin Et"):
                    predicted_label, confidence, probabilities = predict_window(latest_values)
                    st.success(f"Tahmin: {predicted_label}")
                    st.metric("Güven", round(confidence, 4))

                    probability_df = pd.DataFrame(
                        {
                            "sınıf": st.session_state["encoder"].classes_,
                            "olasılık": probabilities,
                        }
                    )
                    st.dataframe(probability_df, use_container_width=True)

        else:
            st.write(
                f"{active_window_size} adet değeri virgülle ayırarak gir. "
                "Örnek: 6.2, 6.3, 6.4"
            )
            manual_text = st.text_area("Sinyal değerleri")

            if st.button("Manuel Değerlerle Tahmin Et"):
                try:
                    manual_values = [
                        float(value.strip())
                        for value in manual_text.replace("\n", ",").split(",")
                        if value.strip()
                    ]

                    if len(manual_values) != active_window_size:
                        st.error(
                            f"Model {active_window_size} değer bekliyor, "
                            f"sen {len(manual_values)} değer girdin."
                        )
                    else:
                        predicted_label, confidence, probabilities = predict_window(manual_values)
                        st.success(f"Tahmin: {predicted_label}")
                        st.metric("Güven", round(confidence, 4))

                        probability_df = pd.DataFrame(
                            {
                                "sınıf": st.session_state["encoder"].classes_,
                                "olasılık": probabilities,
                            }
                        )
                        st.dataframe(probability_df, use_container_width=True)

                except ValueError:
                    st.error("Lütfen sadece sayısal değerler gir.")
