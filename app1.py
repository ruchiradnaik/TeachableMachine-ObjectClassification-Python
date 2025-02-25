import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Title
st.title("Teachable Machine - Object Classification")

# Sidebar Instructions
st.sidebar.header("Steps")
st.sidebar.markdown("1️⃣ Capture Images for Training")
st.sidebar.markdown("2️⃣ Train the Model")
st.sidebar.markdown("3️⃣ Predict Objects Using Webcam")

# Create dataset folder if it doesn't exist
if not os.path.exists("dataset"):
    os.makedirs("dataset")

# Define object categories
categories = st.text_input("Enter object categories (comma-separated):", "Object1, Object2")
categories = [c.strip() for c in categories.split(",")]

# Store dataset paths
# Ensure dataset dictionary is updated dynamically
if "dataset" not in st.session_state:
    st.session_state["dataset"] = {}

# Ensure all categories are in session state
for category in categories:
    if category not in st.session_state["dataset"]:
        st.session_state["dataset"][category] = []



# --------------------------------
# 🔴 Step 1: Capture Images for Training
# --------------------------------
st.header("1️⃣ Capture Images for Training")

selected_category = st.selectbox("Select category to capture images for:", categories)

img_file_buffer = st.camera_input("Capture an image for training")

if img_file_buffer:
    image = Image.open(img_file_buffer)
    
    # Save the image in dataset folder
    category_path = f"dataset/{selected_category}"
    if not os.path.exists(category_path):
        os.makedirs(category_path)

    img_path = os.path.join(category_path, f"{len(os.listdir(category_path))}.jpg")
    image.save(img_path)
    
    st.session_state["dataset"][selected_category].append(img_path)
    st.success(f"✅ Image saved to {img_path}")

# --------------------------------
# 🔵 Step 2: Train the Model
# --------------------------------
st.header("2️⃣ Train the Model")

def load_data():
    images, labels = [], []
    label_map = {category: i for i, category in enumerate(categories)}

    for category in categories:
        category_path = f"dataset/{category}"
        if os.path.exists(category_path):
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                img = Image.open(img_path).resize((128, 128))
                images.append(np.array(img) / 255.0)
                labels.append(label_map[category])

    return np.array(images), np.array(labels)

if st.button("Train Model"):
    images, labels = load_data()
    
    if len(images) == 0:
        st.error("❌ No images found! Please capture images first.")
    else:
        # Define a simple CNN model
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(len(categories), activation='softmax')
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(images, labels, epochs=5)

        model.save("model.h5")
        st.success("🎉 Model trained and saved as `model.h5`")

# --------------------------------
# 🟢 Step 3: Predict Using Webcam
# --------------------------------
st.header("3️⃣ Predict Using Webcam")

if os.path.exists("model.h5"):
    model = tf.keras.models.load_model("model.h5")

    img_file_buffer = st.camera_input("Capture an image for prediction")

    if img_file_buffer:
        image = Image.open(img_file_buffer).resize((128, 128))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_class = categories[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        st.success(f"✅ Predicted: **{predicted_class}** with {confidence:.2f}% confidence")
else:
    st.warning("⚠️ Train the model first before predicting!")

