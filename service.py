from __future__ import annotations

import logging

import numpy as np
import bentoml
from bentoml.io import JSON
from bentoml.io import Text
from bentoml.io import NumpyNdarray
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.middleware.base import BaseHTTPMiddleware

logging.basicConfig(level=logging.DEBUG)

iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])


@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series: np.ndarray) -> np.ndarray:
    """classify a series of numpy array."""
    return iris_clf_runner.run(input_series)


class CustomReadyzMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url == "http://127.0.0.1:3000/readyz":
            return PlainTextResponse("Not ready", status_code=503)
        return await call_next(request)


svc.add_asgi_middleware(CustomReadyzMiddleware)
