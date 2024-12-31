'''
@File_name: 'validate_onnx.py'
@Author: Alexandra Bekou
@Date created: 31-12-24 12:36pm
@Description: Validate model.onnx using onnxruntime
'''
# Import libraries
import onnxruntime as ort
import numpy as np
from ann import sc # Import pre-fitted scaler

# Load the ONNX model
session = ort.InferenceSession("model.onnx")

# Get input and output details
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Prepare a dummy input
input_data = sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])
input_data = np.array(input_data, dtype=np.float32)

# Perform inference
output = session.run([output_name], {input_name: input_data})

# Process the output
print(output) # Expected output: < 0.5
print(output[0] > 0.5) # Expected output : False