'''
@File_name: 'explore_onnx.py'
@Author: Alexandra Bekou
@Date created: 31-12-24 03:07pm
@Description: Explore ONNX architecture of an onnx model with netron
'''
# Import libraries
import onnx
import netron

# Load onnx model from file
model = onnx.load('model.onnx')

# Check model's validity: verifies correct structure
onnx.checker.check_model(model)

# Print the entire model structure
print(onnx.helper.printable_graph(model.graph))

# Detailed model structure breakdown
for node in model.graph.node:
    print(f"Node Name: {node.name}")
    print(f"Operation: {node.op_type}")
    for input_name in node.input:
        print(f"Input: {input_name}")
    for output_name in node.output:
        print(f"Output: {output_name}")
    print("----")

# Visualize ONNX model with Netron
netron.start('model.onnx')