import os
import csv
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from .image_detectors import *
from .image_inference import Resnet50_predict_breed


# preset parameters
dog_names_csv = "data/dog_names.csv"
dog_breeds_path = "dogclassifierapp/static/breeds"
default_breed_image = "static/img/5lrtil.jpg"

saved_best_Resnet50_weights = "data/saved_models/weights.best.Resnet50.hdf5"
saved_model = "data/saved_models/model_Resnet50"


def predict_breed(img_path):
	"""
	Predict dog breed. Detects dog and human and returns their resemblance to closest breed.

	Args:
		img_path: File path to image.

	Returns:
		Prints out if dog or human detected and their closest resemblance to dog breed.
		Or print out neither is detected.
		Image of predicted breed.

	"""
	K.clear_session()

	dog_names = read_dog_names()
	model = load_pretrained_model()

	if dog_detector(img_path) == 1:
		breed = Resnet50_predict_breed(img_path, model, dog_names)
		message = "Dog is detected. Breed of dog is {}.".format(breed)

		breed_image = find_breed_image(breed)

	elif profile_frontal_face_detector(img_path) == 1:
		breed = Resnet50_predict_breed(img_path, model, dog_names)
		message = "Human is detected. The human most resemble {} dog breed.".format(breed)

		breed_image = find_breed_image(breed)
	else:
		message = "Neither dog nor human is detected."

		breed_image = default_breed_image

	return message, breed_image


def find_breed_image(breed):
	"""
	Find breed image from breed name.

	Args:
		breed: Name of breed.
	Returns:
		found_image: File path to image of breed.
	"""
	found_image = None
	for file in os.listdir(dog_breeds_path):
		if breed.lower() in file.lower():
			parent = dog_breeds_path.split("/")[0]
			found_image = os.path.join(dog_breeds_path.replace(parent, ""), file)
			break

	return found_image


def read_dog_names():
	"""
	Read in dog breed names.

	Returns:
		List of dog breeds.
	"""
	with open(dog_names_csv) as f:
		reader = csv.reader(f)
		dog_names = list(reader)[0]

	return dog_names


def load_pretrained_model():
	"""
	Load pretrained model.

	Returns:
		model: model with architecture and weights.
	"""
	model = load_model(saved_model)

	return model

