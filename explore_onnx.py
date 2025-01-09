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

# Detailed model structure breakdown - write ONNX graph to txt file
with open('Model_arch.txt', 'w') as file:

    for node in model.graph.node:
        file.write(f"Node Name: {node.name} \nOperation: {node.op_type}\n")
        for input_name in node.input:
            file.write(f"Input: {input_name}\n")
        for output_name in node.output:
            file.write(f"Output: {output_name}\n")
        file.write("----\n")

# Visualize ONNX model with Netron
netron.start('model.onnx')