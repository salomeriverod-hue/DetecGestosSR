import streamlit as st
import cv2
import numpy as np
from PIL import Image as Image, ImageOps as ImagOps
from tensorflow.keras.models import load_model
import platform

# Mostrar versión de Python
st.write("Versión de Python:", platform.python_version())

# Cargar modelo
@st.cache_resource
def load_my_model():
    return load_model('keras_model.h5')

model = load_my_model()

# Leer labels EXACTOS del archivo
with open("labels.txt", "r") as f:
    labels = [line.strip().split(" ")[1] for line in f.readlines()]

# Crear array de entrada (igual que tu modelo espera)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

st.title("Reconocimiento de Imágenes")

# Imagen de referencia
image = Image.open('OIG5.jpg')
st.image(image, width=350)

with st.sidebar:
    st.subheader("Identifica si la persona es Valentina o Salomé")

# Cámara
img_file_buffer = st.camera_input("Toma una Foto")

if img_file_buffer is not None:

    # Leer imagen
    img = Image.open(img_file_buffer)

    # Redimensionar EXACTO como el modelo
    img = img.resize((224, 224))

    # Convertir a array
    img_array = np.array(img)

    # Normalización EXACTA de Teachable Machine
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1

    data[0] = normalized_image_array

    # Mostrar imagen
    st.image(img, caption="Imagen capturada", use_column_width=True)

    # Predicción
    with st.spinner("Analizando imagen..."):
        prediction = model.predict(data)

    # Elegir la clase con mayor probabilidad (SIN alterar el modelo)
    index = np.argmax(prediction)
    confidence = prediction[0][index]

    # Resultado final (usando labels originales)
    nombre = labels[index]

    # Mostrar SIEMPRE quién es
    st.header(f"Es {nombre} ({confidence:.2f})")

    # Mostrar probabilidades (debug útil)
    st.write("Probabilidades:")
    for i, label in enumerate(labels):
        st.write(f"{label}: {prediction[0][i]:.2f}")

