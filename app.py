import os
import gdown
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Reconocimiento Facial CNN", layout="centered")

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

h_img, w_img = int(img_shape[0]), int(img_shape[1])

st.title("Reconocimiento Facial - CNN Propia (LFW)")
st.write("Sube una imagen de rostro para identificar a la persona.")

archivo = st.file_uploader(
    "Selecciona una imagen",
    type=["jpg", "jpeg", "png"]
)

if archivo is not None:

    img = Image.open(archivo).convert("L")
    img_resized = img.resize((w_img, h_img))

    arr = np.array(img_resized).astype("float32") / 255.0
    arr = arr.reshape(1, h_img, w_img, 1)

    pred = modelo.predict(arr, verbose=0)[0]
    idx_top = np.argsort(pred)[::-1][:5]

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            img,
            caption="Imagen original",
            use_container_width=True
        )

    with col2:
        st.image(
            img_resized,
            caption=f"Procesada ({w_img}x{h_img})",
            use_container_width=True
        )

    st.subheader(f"Predicción: {class_names[idx_top[0]]}")
    st.write(f"Confianza: {pred[idx_top[0]] * 100:.2f}%")

    st.subheader("Top 5 predicciones")

    for i in idx_top:
        st.write(f"{class_names[i]}: {pred[i] * 100:.2f}%")
        st.progress(float(pred[i]))

else:
    st.info("Esperando una imagen...")
