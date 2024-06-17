import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

import keras
import keras_nlp
import numpy as np
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
keras.config.set_floatx("bfloat16")
gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en", load_weights=False)

print(gemma_lm.generate("What is the meaning of life?", max_length=30))

