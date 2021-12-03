import cv2
from .image_inference import ResNet50_predict_labels

# Path to pre-trained XMLs
frontal_face_xml = "data/haarcascades/haarcascade_frontalface_alt.xml"
frontal_face2_xml = "data/haarcascades/haarcascade_frontalface_alt2.xml"
profile_face_xml = "data/haarcascades/haarcascade_profileface.xml"


def face_detector(img_path):
    """
    Human face detector.

    Args:
        img_path: File path to image.

    Return:
        "True" if face is detected. Otherwise "False".
    """
    face_cascade = cv2.CascadeClassifier(frontal_face_xml)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    return len(faces) > 0


def profile_frontal_face_detector(img_path):
    """
    Detects frontal and profile human faces.

    Args:
        img_path: image file paths.

    Returns:
        is_face: binary of human face or not.
    """
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(frontal_face2_xml)
    faces = face_cascade.detectMultiScale(gray)

    profileface_cascade = cv2.CascadeClassifier(profile_face_xml)
    profilefaces = profileface_cascade.detectMultiScale(gray)

    is_face = False
    if len(faces) > 0 or len(profilefaces):
        is_face = True

    return is_face


def dog_detector(img_path):
    """
    ResNet50 image classifier to detect if image is a dog.
    According to dictionary, https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a, categories corresponding to dogs
    appear in an uninterrupted sequence and correspond to dictionary keys 151-268, inclusive, to include all
    categories from `'Chihuahua'` to `'Mexican hairless'`.

    Args:
        img_path: File path to image.

    Returns:
        "True" if a dog is detected in the image stored at img_path.
    """
    prediction = ResNet50_predict_labels(img_path)
    return (prediction <= 268) & (prediction >= 151)
