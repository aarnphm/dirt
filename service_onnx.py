import bentoml

from bentoml.io import JSON
from bentoml.io import NumpyNdarray

runner = bentoml.onnx.get("onnx_iris:latest").to_runner()

svc = bentoml.Service("onnx_iris", runners=[runner])


@svc.api(input=NumpyNdarray(), output=JSON())
def classify(input_array):
    return runner.run.run(input_array)
