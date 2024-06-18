import tensorflow
from tensorflow import keras
import keras_nlp
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import os


api = KaggleApi()
api.authenticate()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"
keras.config.set_floatx("float16")
gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_1.1_instruct_2b_en", load_weights=False)
print(f'GemmaCasualLM generate: {gemma_lm.generate("What is the meaning of life?", max_length=30)}')

gpu_devices = tensorflow.config.list_physical_devices('GPU')
print(tensorflow.config.experimental.get_device_details(gpu_devices[0]))
print(tensorflow.config.experimental.get_memory_info(f'{gpu_devices[0].device_type}:0'))