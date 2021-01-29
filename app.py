from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits
from image_recognition import process_image

# Specify canvas parameters in application
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
drawing_mode = st.sidebar.selectbox('Drawing Tool', ['freedraw'])
realtime_update = st.sidebar.checkbox("Update in realtime", True)

st.write("Draw a number between 0 and 9")
# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    update_streamlit=realtime_update,
    height=150,
    drawing_mode=drawing_mode,
    key="canvas",
)


# Method to train model with necessary image data
def train_model(data, target):
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(data, target)
    return model

# Load MNIST digit dataset
digits = load_digits()

# Train the model
classifier = train_model(digits.data, digits.target)

# Process the input image
x = process_image(canvas_result.image_data)

# x = x.reshape((-1,2))

if x is not None:
    prediction = classifier.predict(x)[0]



# Let the model predict the number drawn on the canvas
if st.button('Predict'):
    st.write('You predicted {}'.format(prediction))

# Do something interesting with the image data and paths
# if canvas_result.image_data is not None:
#     st.image(canvas_result.image_data)