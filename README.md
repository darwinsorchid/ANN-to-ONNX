# Export TensorFlow ANN to ONNX format

## Description
> Use `tf2onnx` library to convert a simple ANN model to ONNX format. 

> Download models from [ONNX Model Zoo](https://onnx.ai/models/).

> Visualize ONNX models architecture with `netron`.

## Dependencies
- `Python 10.0`

- `Tensorflow 2.15.0`

- After saving model in SavedModel format run `python -m tf2onnx.convert --saved-model saved_model --output model.onnx` in terminal.
