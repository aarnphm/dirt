from typing import List
  
import bentoml
import numpy as np
import torch
from bentoml.io import JSON

runner = bentoml.pytorch.get("yolov5:latest").to_runner()
service = bentoml.Service("yolov5", runners=[runner])


@service.api(input=JSON(), output=JSON())
def predict(input_dict):
    filename = input_dict['filename']
    img = input_dict['image_content']
    input_img = torch.from_numpy(np.asarray(img[0]))
    predictions = runner.run_batch(input_img)
    return predictions
    
