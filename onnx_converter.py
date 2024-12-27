# Import libraries
import tensorflow as tf
import onnx
import tf2onnx

# Load model 
loaded_keras_model = tf.keras.models.load_model('saved_model.h5')

# Model summary 
loaded_keras_model.summary()

# Convert loaded model to onnx format
onnx_model, _ = tf2onnx.convert.from_keras(loaded_keras_model)

# Save onnx_model
onnx.save(onnx_model, 'model.onnx')