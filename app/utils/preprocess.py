from PIL import Image
import numpy as np
from io import BytesIO

def preprocess_image(image_data):
    image = Image.open(BytesIO(image_data)).convert("RGB")
    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array
