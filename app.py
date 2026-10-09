import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt


# =========================
# LOAD DATA
# =========================

df = pd.read_csv("Credit_Score_Clean.csv")

feature_data = np.load(
    "features_relief_5.npy",
    allow_pickle=True
).item()

feature_data_up = np.load(
    "features_relief_5_up.npy",
    allow_pickle=True
).item()

X_test = feature_data["X_test"]
y_test = feature_data["y_test"] + 1

X_test_up = feature_data_up["X_test"]
y_test_up = feature_data_up["y_test"] + 1


# =========================
# LOAD MODELS
# =========================

models = {
    "Linear SVM": joblib.load("svm.pkl"),
    "Cubic KNN": joblib.load("knn.pkl"),
    "LDA": joblib.load("lda.pkl"),
    "Gaussian NB": joblib.load("nb.pkl")
}

# =========================
# UI
# =========================

st.title("Credit Score Classification")
tab1, tab2 = st.tabs([
    "Baseline Model",
    "Improved Model"
])

with tab1:

    st.write("Demo using 5 Relief Features")


    # chọn model
    model_name = st.selectbox(
        "Choose model",
        list(models.keys())
    )

    model = models[model_name]

    # chọn sample
    idx = st.number_input(
        "Enter test sample index (max 9513)",
        min_value=0,
        max_value=len(X_test)-1,
        value=0,
        step=1
    )

    from utils import row_to_image

    row = df.iloc[idx]

    img = row_to_image(
        row.drop("Credit_Score")
    )

    st.subheader("Generated Grayscale Image")

    fig, ax = plt.subplots(figsize=(6,6))

    ax.imshow(
        img,
        cmap='gray',
        interpolation='nearest'
    )

    ax.set_title("14x14 Grayscale Image")

    ax.axis('off')

    st.pyplot(fig)

    # feature
    sample = X_test[idx].reshape(1, -1)
    true_label = y_test[idx]
    # predict
    if hasattr(model, "predict_proba"):

        probs = model.predict_proba(sample)[0]

        pred = model.classes_[np.argmax(probs)]

    else:

        pred = model.predict(sample)[0]

    # probability
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(sample)[0]


    # =========================
    # SHOW RESULT
    # =========================

    label_map = {
        1: "GOOD",
        2: "POOR",
        3: "STANDARD"
    }

    st.subheader("Prediction")

    st.success(label_map[pred])


    st.subheader("Ground Truth")

    st.info(label_map[true_label])
    if pred == true_label:
        st.success("Prediction Correct")
    else:
        st.error("Prediction Incorrect")

    # =========================
    # SHOW FEATURE
    # =========================

    st.subheader("5 Features")

    st.write(sample)

    # =========================
    # PROBABILITY
    # =========================

    if hasattr(model, "predict_proba"):

        fig, ax = plt.subplots()

        class_names = []

        for c in model.classes_:

            if c == 1:
                class_names.append("Good")

            elif c == 2:
                class_names.append("Poor")

            elif c == 3:
                class_names.append("Standard")

        ax.bar(
            class_names,
            probs
        )

        st.pyplot(fig)


with tab2:

    st.header("Improved Architecture")

    st.write(
        """
        Improved NewFC Layers:
        - DenseNet201: 1920 → 128 → 3
        - ResNet18: 512 → 64 → 3
        """
    )

    improved_models = {

        "Linear SVM": joblib.load("svm_up.pkl"),

        "Cubic KNN": joblib.load("knn_up.pkl"),

        "LDA": joblib.load("lda_up.pkl"),

        "Gaussian NB": joblib.load("nb_up.pkl")
    }

    model_name_up = st.selectbox(
        "Choose improved model",
        list(improved_models.keys()),
        key="improved_model"
    )

    model_up = improved_models[model_name_up]

    sample_up = X_test_up[idx].reshape(1, -1)

    true_label_up = y_test_up[idx]

    if hasattr(model_up, "predict_proba"):

        probs_up = model_up.predict_proba(sample_up)[0]

        pred_up = model_up.classes_[
            np.argmax(probs_up)
        ]

    else:

        pred_up = model_up.predict(sample_up)[0]

    label_map = {
        1: "GOOD",
        2: "POOR",
        3: "STANDARD"
    }

    st.subheader("Prediction")

    st.success(label_map[pred_up])

    st.subheader("Ground Truth")

    st.info(label_map[true_label_up])

    if pred_up == true_label_up:

        st.success("Prediction Correct")

    else:

        st.error("Prediction Incorrect")

    st.subheader("5 Improved Features")

    st.write(sample_up)

    if hasattr(model_up, "predict_proba"):

        st.subheader("Prediction Probability")

        fig, ax = plt.subplots()

        class_names_up = []

        for c in model_up.classes_:

            if c == 1:
                class_names_up.append("Good")

            elif c == 2:
                class_names_up.append("Poor")

            elif c == 3:
                class_names_up.append("Standard")

        ax.bar(
            class_names_up,
            probs_up
        )

        st.pyplot(fig)


