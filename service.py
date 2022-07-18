from __future__ import annotations

import typing as t
import bentoml
from bentoml.io import Text, JSON
import inspect


runner = bentoml.transformers.get("tiny_random_bert").to_runner()
svc = bentoml.Service("batch_pipeline", runners=[runner])


@svc.api(input=Text(), output=JSON())
async def classify(input_series: list[str]) -> dict[str, t.Any]:
    """classify a series of numpy array."""
    res = inspect.get_annotations(
        runner.runnable_class.bentoml_runnable_methods__["__call__"].func
    )
    print(res)
    return await runner.async_run(*input_series)
