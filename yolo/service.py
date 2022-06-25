import logging
import bentoml
import torch
import numpy as np
from bentoml.io import JSON

runner = bentoml.pytorch.get("yolov5:latest").to_runner()
service = bentoml.Service("yolov5", runners=[runner])


@service.api(input=JSON(), output=JSON())
async def predict(input_dict):
    img = input_dict["image_content"]
    input_img = torch.from_numpy(np.asarray(img[0]))
    predictions = await runner.async_run(input_img)
    return predictions
