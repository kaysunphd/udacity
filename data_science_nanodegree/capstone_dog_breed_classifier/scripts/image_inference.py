import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.preprocessing import image
from tqdm import tqdm
from .extract_bottleneck_features import *


def path_to_tensor(img_path):
    """
    Convert image to 4D tensor (number, x-dimension, y-dimension, channels)

    Args:
        img_path: File path to image.

    Returns:
        4D tensor.
    """
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def ResNet50_predict_labels(img_path):
    """
    Returns prediction vector for image located at img_path.

    Args:
        img_path: File path to image.

    Returns:
        Prediction vector for image.
    """
    img = preprocess_input(path_to_tensor(img_path))
    model = ResNet50(weights='imagenet')

    return np.argmax(model.predict(img))


def Resnet50_predict_breed(img_path, model, reference_list_names):
    """
    Predict dog breed from model.

    Args:
        img_path: File path to image
        model: Trained model for inference.
        reference_list_names:  List of names referenced from model.

    Returns:
        Dog breed predicted by model
    """
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))

    # obtain predicted vector
    predicted_vector = model.predict(bottleneck_feature)

    # return dog breed that is predicted by the model
    return reference_list_names[np.argmax(predicted_vector)].split(".")[-1]
