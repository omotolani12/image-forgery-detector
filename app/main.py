import tensorflow as tf
import pennylane as qml
import numpy as np
from fastapi import FastAPI, File, UploadFile
from app.utils.preprocess import preprocess_image
import uvicorn

# Quantum setup
n_qubits = 16
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_circuit(inputs):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights=np.random.random((3, n_qubits)), wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumLayer(tf.keras.layers.Layer):
    def __init__(self, n_qubits=16, trainable=True, **kwargs):
        super().__init__(trainable=trainable, **kwargs)
        self.n_qubits = n_qubits

    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float32)
        q_out = tf.map_fn(lambda x: tf.convert_to_tensor(quantum_circuit(x), dtype=tf.float32), inputs)
        return tf.reshape(q_out, (-1, self.n_qubits))

    def get_config(self):
        config = super().get_config()
        config.update({"n_qubits": self.n_qubits})
        return config

model = tf.keras.models.load_model("app/model/quantum_forgery_detector.h5", custom_objects={"QuantumLayer": QuantumLayer})

app = FastAPI()

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image_array = preprocess_image(image_data)
        prediction = model.predict(image_array)
        is_forged = bool(prediction[0][0] > 0.5)
        return {
            "filename": file.filename,
            "prediction": "Fake" if is_forged else "Real",
            "confidence": float(prediction[0][0])
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)