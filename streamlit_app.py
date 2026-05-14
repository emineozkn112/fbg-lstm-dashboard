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
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    LSTM,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling1D,
)
from tensorflow.keras.models import Sequential, load_model


st.set_page_config(page_title="FBG LSTM + CNN Dashboard", layout="wide")


def get_project_dir():
    script_dir = Path(__file__).resolve().parent
    candidates = [script_dir, script_dir.parent, Path.cwd()]
    for candidate in candidates:
        if (candidate / "fbg_filtered_dataset.csv").exists():
            return candidate
    return script_dir


PROJECT_DIR = get_project_dir()
MODEL_DIR = PROJECT_DIR / "models"
DEFAULT_CSV = PROJECT_DIR / "fbg_filtered_dataset.csv"

LSTM_ARTIFACTS = {
    "model": "fbg_lstm_model.keras",
    "scaler": "scaler.joblib",
    "encoder": "label_encoder.joblib",
    "config": "model_config.json",
}

CNN_ARTIFACTS = {
    "model": "fbg_1dcnn_model.keras",
    "scaler": "cnn_scaler.joblib",
    "encoder": "cnn_label_encoder.joblib",
    "config": "cnn_model_config.json",
}


def artifact_path(filename):
    model_path = MODEL_DIR / filename
    flat_path = PROJECT_DIR / filename
    if model_path.exists():
        return model_path
    if flat_path.exists():
        return flat_path
    return model_path


def artifacts_exist(artifact_names):
    return all(artifact_path(filename).exists() for filename in artifact_names.values())


def load_dataset(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file), uploaded_file.name
    return pd.read_csv(DEFAULT_CSV), str(DEFAULT_CSV)


def validate_columns(df, feature_column, needs_label):
    required = [feature_column]
    if needs_label:
        required.append("label")
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError("Eksik sütunlar: " + ", ".join(missing))


def make_windows(df, feature_column, label_column, window_size, stride=1):
    x_values = pd.to_numeric(df[feature_column], errors="coerce")

    if label_column and label_column in df.columns:
        valid_mask = x_values.notna() & df[label_column].notna()
    else:
        valid_mask = x_values.notna()

    series = x_values[valid_mask].astype(float).values
    labels = None
    if label_column and label_column in df.columns:
        labels = df.loc[valid_mask, label_column].astype(str).str.strip().values

    if len(series) < window_size:
        raise ValueError("Veri uzunluğu pencere boyutundan küçük.")

    x_windows = []
    y_windows = []

    for start in range(0, len(series) - window_size + 1, stride):
        end = start + window_size
        x_windows.append(series[start:end])

        if labels is not None:
            window_labels = labels[start:end]
            dominant_label = Counter(window_labels).most_common(1)[0][0]
            y_windows.append(dominant_label)

    x_windows = np.array(x_windows, dtype=np.float32)[..., np.newaxis]

    if labels is None:
        return x_windows, None

    return x_windows, np.array(y_windows)


def scale_lstm_train_test(x_train, x_test):
    scaler = StandardScaler()
    feature_count = x_train.shape[2]
    x_train_2d = x_train.reshape(-1, feature_count)
    x_test_2d = x_test.reshape(-1, feature_count)
    x_train_scaled = scaler.fit_transform(x_train_2d).reshape(x_train.shape)
    x_test_scaled = scaler.transform(x_test_2d).reshape(x_test.shape)
    return x_train_scaled, x_test_scaled, scaler


def scale_cnn_full_windows(x_windows):
    scaler = MinMaxScaler()
    feature_count = x_windows.shape[2]
    x_2d = x_windows.reshape(-1, feature_count)
    x_scaled = scaler.fit_transform(x_2d).reshape(x_windows.shape)
    return x_scaled, scaler


def build_lstm_model(window_size, feature_count, class_count):
    model = Sequential(
        [
            Input(shape=(window_size, feature_count)),
            LSTM(64),
            Dropout(0.25),
            Dense(32, activation="relu"),
            Dense(class_count, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn_model(window_size, feature_count, class_count):
    model = Sequential(
        [
            Input(shape=(window_size, feature_count)),
            Conv1D(16, 3, activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.25),
            Conv1D(32, 3, activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.30),
            Flatten(),
            Dense(64, activation="relu"),
            Dropout(0.35),
            Dense(class_count, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_artifacts(artifact_names, model, scaler, encoder, config):
    MODEL_DIR.mkdir(exist_ok=True)
    model.save(MODEL_DIR / artifact_names["model"])
    joblib.dump(scaler, MODEL_DIR / artifact_names["scaler"])
    joblib.dump(encoder, MODEL_DIR / artifact_names["encoder"])
    (MODEL_DIR / artifact_names["config"]).write_text(
        json.dumps(config, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )


def load_artifacts(artifact_names):
    model = load_model(artifact_path(artifact_names["model"]))
    scaler = joblib.load(artifact_path(artifact_names["scaler"]))
    encoder = joblib.load(artifact_path(artifact_names["encoder"]))
    config = json.loads(artifact_path(artifact_names["config"]).read_text(encoding="utf-8"))
    return model, scaler, encoder, config


def store_active_model(model_type, model, scaler, encoder, config):
    st.session_state["active_model_type"] = model_type
    st.session_state["active_model"] = model
    st.session_state["active_scaler"] = scaler
    st.session_state["active_encoder"] = encoder
    st.session_state["active_config"] = config


def active_model_ready():
    return all(
        key in st.session_state
        for key in ["active_model_type", "active_model", "active_scaler", "active_encoder", "active_config"]
    )


def predict_window(values):
    model = st.session_state["active_model"]
    scaler = st.session_state["active_scaler"]
    encoder = st.session_state["active_encoder"]
    config = st.session_state["active_config"]

    window_size = int(config["window_size"])
    feature_count = int(config["feature_count"])
    values = np.array(values, dtype=np.float32).reshape(window_size, feature_count)
    values_2d = values.reshape(-1, feature_count)
    values_scaled = scaler.transform(values_2d).reshape(1, window_size, feature_count)

    probabilities = model.predict(values_scaled, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_label = encoder.inverse_transform([predicted_index])[0]
    confidence = float(probabilities[predicted_index])
    return predicted_label, confidence, probabilities


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


def plot_history(history, title_prefix):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="Eğitim Doğruluğu")
    axes[0].plot(history.history["val_accuracy"], label="Doğrulama Doğruluğu")
    axes[0].set_title(f"{title_prefix} Doğruluk Grafiği")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Eğitim Kaybı")
    axes[1].plot(history.history["val_loss"], label="Doğrulama Kaybı")
    axes[1].set_title(f"{title_prefix} Kayıp Grafiği")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True)
    axes[1].legend()

    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm, class_names, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(cm, display_labels=class_names)
    display.plot(cmap="Blues", values_format="d", ax=ax)
    ax.set_title(title)
    return fig


def show_results(model_name, test_accuracy, test_loss, window_count, history, cm, encoder, y_test, y_pred):
    st.success(f"{model_name} modeli eğitildi ve kaydedildi.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Test Accuracy", round(float(test_accuracy), 4))
    col2.metric("Test Loss", round(float(test_loss), 4))
    col3.metric("Pencere Sayısı", int(window_count))

    st.pyplot(plot_history(history, model_name))
    st.pyplot(plot_confusion_matrix(cm, encoder.classes_, f"{model_name} Confusion Matrix"))

    report = classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)


st.title("FBG Sensör Verisi ile LSTM + CNN Hasar Tespiti Dashboard")

st.sidebar.header("Veri ve Model")
uploaded_file = st.sidebar.file_uploader("CSV dosyası yükle", type=["csv"])

try:
    df, data_source = load_dataset(uploaded_file)
except FileNotFoundError:
    st.error("Varsayılan CSV bulunamadı. Lütfen sol menüden CSV dosyası yükle.")
    st.stop()

numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
feature_candidates = [column for column in numeric_columns if column != "time"]

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

has_label = "label" in df.columns

st.sidebar.subheader("LSTM Ayarları")
lstm_window_size = st.sidebar.slider("LSTM pencere boyutu", 10, 128, 32, 2)
lstm_stride = st.sidebar.slider("LSTM stride", 1, 32, 8, 1)
lstm_epochs = st.sidebar.slider("LSTM epoch sayısı", 5, 60, 20, 5)

st.sidebar.subheader("CNN Ayarları")
cnn_window_size = st.sidebar.slider("CNN pencere boyutu", 16, 128, 32, 8)
cnn_stride = st.sidebar.slider("CNN stride", 1, 32, 8, 1)
cnn_epochs = st.sidebar.slider("CNN epoch sayısı", 5, 80, 50, 5)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Satır Sayısı", len(df))
col2.metric("Sütun Sayısı", len(df.columns))
col3.metric("Sinyal Sütunu", feature_column)
col4.metric("Etiket Var mı?", "Evet" if has_label else "Hayır")

st.caption(f"Kullanılan veri: {data_source}")

overview_tab, lstm_tab, cnn_tab, live_tab = st.tabs(
    ["Veri İnceleme", "LSTM Eğitimi", "CNN Eğitimi", "Canlı Tahmin"]
)

with overview_tab:
    st.subheader("Veri Ön İzleme")
    preview_rows = st.slider("Gösterilecek satır sayısı", 5, min(len(df), 100), 10)
    st.dataframe(df.head(preview_rows), use_container_width=True)

    st.subheader("Sinyal Grafiği")
    st.pyplot(plot_signal(df, feature_column))

    if has_label:
        st.subheader("Etiket Dağılımı")
        label_counts = df["label"].value_counts()
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

with lstm_tab:
    st.subheader("LSTM Model Eğitimi")
    st.write(
        "LSTM, zaman serisindeki ardışık ölçümler arasındaki ilişkiyi öğrenir. "
        "Bu bölüm mevcut panelindeki RNN/LSTM tabanlı hasar sınıflandırma akışıdır."
    )

    lstm_train_disabled = not has_label or len(df) <= lstm_window_size
    if lstm_train_disabled:
        st.warning("LSTM eğitimi için CSV içinde label sütunu olmalı ve veri pencere boyutundan uzun olmalı.")

    if st.button("LSTM Modelini Eğit ve Kaydet", disabled=lstm_train_disabled):
        validate_columns(df, feature_column, needs_label=True)
        x_windows, y_text = make_windows(df, feature_column, "label", lstm_window_size, stride=lstm_stride)

        encoder = LabelEncoder()
        y_numeric = encoder.fit_transform(y_text)

        x_train, x_test, y_train, y_test = train_test_split(
            x_windows,
            y_numeric,
            test_size=0.2,
            random_state=42,
            stratify=y_numeric,
        )

        x_train_scaled, x_test_scaled, scaler = scale_lstm_train_test(x_train, x_test)

        model = build_lstm_model(
            window_size=x_train_scaled.shape[1],
            feature_count=x_train_scaled.shape[2],
            class_count=len(encoder.classes_),
        )

        with st.spinner("LSTM modeli eğitiliyor..."):
            history = model.fit(
                x_train_scaled,
                y_train,
                epochs=lstm_epochs,
                batch_size=32,
                validation_split=0.2,
                verbose=0,
            )

        test_loss, test_accuracy = model.evaluate(x_test_scaled, y_test, verbose=0)
        probabilities = model.predict(x_test_scaled, verbose=0)
        y_pred = np.argmax(probabilities, axis=1)
        cm = confusion_matrix(y_test, y_pred)

        config = {
            "model_type": "LSTM",
            "feature_column": feature_column,
            "label_column": "label",
            "window_size": int(lstm_window_size),
            "stride": int(lstm_stride),
            "feature_count": int(x_train_scaled.shape[2]),
            "class_names": encoder.classes_.tolist(),
            "source": str(data_source),
            "scaling": "StandardScaler",
        }

        save_artifacts(LSTM_ARTIFACTS, model, scaler, encoder, config)
        store_active_model("LSTM", model, scaler, encoder, config)
        show_results("LSTM", test_accuracy, test_loss, len(x_windows), history, cm, encoder, y_test, y_pred)

    st.subheader("Kayıtlı LSTM Modeli")
    if st.button("Kayıtlı LSTM Modelini Yükle", disabled=not artifacts_exist(LSTM_ARTIFACTS)):
        model, scaler, encoder, config = load_artifacts(LSTM_ARTIFACTS)
        store_active_model("LSTM", model, scaler, encoder, config)
        st.success("Kayıtlı LSTM modeli aktif edildi.")

    if not artifacts_exist(LSTM_ARTIFACTS):
        st.info("Henüz kayıtlı LSTM modeli bulunamadı. Önce LSTM modelini eğitip kaydet.")

with cnn_tab:
    st.subheader("1D-CNN Model Eğitimi")
    st.write(
        "CNN bölümü, yüklediğin zip paketindeki Zeynep'in 1D-CNN mimarisinden alınmıştır. "
        "Conv1D katmanları sinyal üzerindeki yerel örüntüleri yakalar; bu yüzden LSTM ile "
        "pattern recognition açısından karşılaştırma yapmak için uygundur."
    )

    cnn_train_disabled = not has_label or len(df) <= cnn_window_size
    if cnn_train_disabled:
        st.warning("CNN eğitimi için CSV içinde label sütunu olmalı ve veri pencere boyutundan uzun olmalı.")

    if st.button("CNN Modelini Eğit ve Kaydet", disabled=cnn_train_disabled):
        validate_columns(df, feature_column, needs_label=True)
        x_windows, y_text = make_windows(df, feature_column, "label", cnn_window_size, stride=cnn_stride)
        x_scaled, scaler = scale_cnn_full_windows(x_windows)

        encoder = LabelEncoder()
        y_numeric = encoder.fit_transform(y_text)

        x_train, x_test, y_train, y_test = train_test_split(
            x_scaled,
            y_numeric,
            test_size=0.2,
            random_state=42,
            stratify=y_numeric,
        )
        x_train, x_val, y_train, y_val = train_test_split(
            x_train,
            y_train,
            test_size=0.2,
            random_state=42,
            stratify=y_train,
        )

        model = build_cnn_model(
            window_size=x_scaled.shape[1],
            feature_count=x_scaled.shape[2],
            class_count=len(encoder.classes_),
        )

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5),
        ]

        with st.spinner("CNN modeli eğitiliyor..."):
            history = model.fit(
                x_train,
                y_train,
                validation_data=(x_val, y_val),
                epochs=cnn_epochs,
                batch_size=8,
                callbacks=callbacks,
                verbose=0,
            )

        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        probabilities = model.predict(x_test, verbose=0)
        y_pred = np.argmax(probabilities, axis=1)
        cm = confusion_matrix(y_test, y_pred)

        config = {
            "model_type": "1D-CNN",
            "feature_column": feature_column,
            "label_column": "label",
            "window_size": int(cnn_window_size),
            "stride": int(cnn_stride),
            "feature_count": int(x_scaled.shape[2]),
            "class_names": encoder.classes_.tolist(),
            "source": str(data_source),
            "scaling": "MinMaxScaler",
        }

        save_artifacts(CNN_ARTIFACTS, model, scaler, encoder, config)
        store_active_model("1D-CNN", model, scaler, encoder, config)
        show_results("1D-CNN", test_accuracy, test_loss, len(x_windows), history, cm, encoder, y_test, y_pred)

    st.subheader("Kayıtlı CNN Modeli")
    if st.button("Kayıtlı CNN Modelini Yükle", disabled=not artifacts_exist(CNN_ARTIFACTS)):
        model, scaler, encoder, config = load_artifacts(CNN_ARTIFACTS)
        store_active_model("1D-CNN", model, scaler, encoder, config)
        st.success("Kayıtlı CNN modeli aktif edildi.")

    if not artifacts_exist(CNN_ARTIFACTS):
        st.info("Henüz kayıtlı CNN modeli bulunamadı. Önce CNN modelini eğitip kaydet.")

with live_tab:
    st.subheader("Canlı Tahmin Paneli")

    load_col1, load_col2 = st.columns(2)
    with load_col1:
        if st.button("LSTM'i Tahmin İçin Aktif Et", disabled=not artifacts_exist(LSTM_ARTIFACTS)):
            model, scaler, encoder, config = load_artifacts(LSTM_ARTIFACTS)
            store_active_model("LSTM", model, scaler, encoder, config)
            st.success("LSTM modeli aktif edildi.")

    with load_col2:
        if st.button("CNN'i Tahmin İçin Aktif Et", disabled=not artifacts_exist(CNN_ARTIFACTS)):
            model, scaler, encoder, config = load_artifacts(CNN_ARTIFACTS)
            store_active_model("1D-CNN", model, scaler, encoder, config)
            st.success("CNN modeli aktif edildi.")

    if not active_model_ready():
        st.warning("Tahmin için önce LSTM veya CNN modelini eğit ya da kayıtlı modeli yükle.")
    else:
        model_type = st.session_state["active_model_type"]
        config = st.session_state["active_config"]
        active_feature = config["feature_column"]
        active_window_size = int(config["window_size"])

        st.write("Aktif model:")
        st.json(config)

        if active_feature not in df.columns:
            st.error(f"Aktif model {active_feature} sütununu bekliyor, ancak bu CSV içinde yok.")
        else:
            st.write(
                f"{model_type} modeli son {active_window_size} ölçümü kullanarak tahmin yapacak. "
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
                                "sınıf": st.session_state["active_encoder"].classes_,
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
                                    "sınıf": st.session_state["active_encoder"].classes_,
                                    "olasılık": probabilities,
                                }
                            )
                            st.dataframe(probability_df, use_container_width=True)

                    except ValueError:
                        st.error("Lütfen sadece sayısal değerler gir.")
