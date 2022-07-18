import logging

import bentoml

logging.basicConfig(level=logging.WARN)

from transformers import pipeline

TINY_TEXT_MODEL = "hf-internal-testing/tiny-random-distilbert"

from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(TINY_TEXT_MODEL)

model = AutoModelForSequenceClassification.from_pretrained(TINY_TEXT_MODEL)


if __name__ == "__main__":

    transformers_model = bentoml.transformers.save_model(
        "tiny_random_bert",
        pipeline(task="text-classification", model=model, tokenizer=tokenizer),
        signatures={"__call__": {"batchable": True}},
    )
