import tensorflow

def load_weight(model_path):
    model = tensorflow.keras.models.get_weights(model_path)
    return model

load_weight("mmodelfinal.h5")