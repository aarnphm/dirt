from __future__ import annotations

import bentoml
from bentoml.io import JSON
from bentoml.io import NumpyNdarray
from numpy.typing import ArrayLike


bento_model = bentoml.sklearn.get("iris_clf:latest")


class SpamDetectionRunnable(bentoml.Runnable):
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        # load the model instance
        self.classifier = bentoml.sklearn.load_model(bento_model)

    @bentoml.Runnable.method(batchable=False)
    def is_spam(self, input_data: ArrayLike) -> ArrayLike:
        return self.classifier.predict(input_data)


spam_detection_runner = bentoml.Runner(SpamDetectionRunnable, models=[bento_model])


svc = bentoml.Service("spam_detector", runners=[spam_detection_runner])


@svc.api(input=NumpyNdarray(), output=JSON())
def analysis(input_text):
    return {"res": spam_detection_runner.is_spam.run(input_text)}
