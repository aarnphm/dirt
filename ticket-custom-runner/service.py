from __future__ import annotations

import bentoml
from bentoml.io import NumpyNdarray, JSON


class spamdetectionrunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ()
    SUPPORTS_CPU_MULTI_THREADING = False

    def __init__(self):
        # load the model back:
        self.classifier = bentoml.sklearn.load_model("iris_clf:latest")

    @bentoml.Runnable.method(batchable=False)
    def is_spam(self, input_):
        return self.classifier.predict(input_)


spam_detection_runner = bentoml.Runner(
    spamdetectionrunnable, models=[bentoml.models.get("iris_clf:latest")]
)

svc = bentoml.Service("spam_detector", runners=[spam_detection_runner])


@svc.api(input=NumpyNdarray(), output=JSON())
def analysis(input_text):
    return {"res": spam_detection_runner.is_spam.run(input_text)}
