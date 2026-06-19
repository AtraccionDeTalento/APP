import os
import gdown
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Reconocimiento Facial CNN",
    layout="centered"
)

MODEL_FILE = "modelo_lfw_cnn.keras"


@st.cache_resource
def cargar_modelo():

    if not os.path.exists(MODEL_FILE):

        file_id = "1FmOd9lUJZ5TFLgZ4e7smuz45C9Rgu56z"
        url = f"https://drive.google.com/uc?id={file_id}"

        with st.spinner("Descargando modelo CNN..."):
            gdown.download(url, MODEL_FILE, quiet=False)

    modelo = load_model(MODEL_FILE)
    class_names = np.load("class_names.npy", allow_pickle=True)
    img_shape = np.load("img_shape.npy")

    return modelo, class_names, img_shape


modelo, class_names, img_shape = cargar_modelo()

h_img = int(img_shape[0])
w_img = int(img_shape[1])


def procesar_imagen(img_file):

    img = Image.open(img_file).convert("L")
    img_resized = img.resize((w_img, h_img))

    arr = np.array(img_resized).astype("float32") / 255.0
    arr = arr.reshape(1, h_img, w_img, 1)

    pred = modelo.predict(arr, verbose=0)[0]

    idx_top = np.argsort(pred)[::-1][:5]
    clase = class_names[idx_top[0]]
    confianza = float(pred[idx_top[0]])

    return img, img_resized, pred, idx_top, clase, confianza


st.title("Reconocimiento Facial CNN")

modo = st.radio(
    "Seleccione una opción",
    [
        "Reconocimiento Facial",
        "Comparación de Rostros"
    ]
)

# ======================================================
# RECONOCIMIENTO
# ======================================================

if modo == "Reconocimiento Facial":

    archivo = st.file_uploader(
        "Seleccione una imagen",
        type=["jpg", "jpeg", "png"],
        key="reconocimiento"
    )

    if archivo is not None:

        img, img_resized, pred, idx_top, clase, confianza = procesar_imagen(archivo)

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                img,
                caption="Imagen Original",
                use_container_width=True
            )

        with col2:
            st.image(
                img_resized,
                caption="Imagen Procesada",
                use_container_width=True
            )

        st.success(f"Predicción: {clase}")
        st.write(f"Confianza: {confianza * 100:.2f}%")

        st.subheader("Top 5 Predicciones")

        for i in idx_top:
            st.write(
                f"{class_names[i]}: {pred[i] * 100:.2f}%"
            )
            st.progress(float(pred[i]))

    else:
        st.info("Esperando imagen...")


# ======================================================
# COMPARACION
# ======================================================

elif modo == "Comparación de Rostros":

    archivo1 = st.file_uploader(
        "Imagen de Referencia",
        type=["jpg", "jpeg", "png"],
        key="img1"
    )

    archivo2 = st.file_uploader(
        "Imagen a Comparar",
        type=["jpg", "jpeg", "png"],
        key="img2"
    )

    if archivo1 is not None and archivo2 is not None:

        img1, _, pred1, _, clase1, conf1 = procesar_imagen(archivo1)
        img2, _, pred2, _, clase2, conf2 = procesar_imagen(archivo2)

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                img1,
                caption="Imagen 1",
                use_container_width=True
            )

        with col2:
            st.image(
                img2,
                caption="Imagen 2",
                use_container_width=True
            )

        st.subheader("Resultado")

        st.write(
            f"Imagen 1: {clase1} ({conf1 * 100:.2f}%)"
        )

        st.write(
            f"Imagen 2: {clase2} ({conf2 * 100:.2f}%)"
        )

        if clase1 == clase2:

            similitud = (
                min(conf1, conf2) /
                max(conf1, conf2)
            ) * 100

            st.success(
                f"Posiblemente la misma persona. Similitud estimada: {similitud:.2f}%"
            )

        else:

            st.error(
                "Personas diferentes según la clasificación del modelo."
            )

    else:
        st.info(
            "Suba ambas imágenes para realizar la comparación."
        )
