# Imports
from pathlib import Path
import face_recognition
import pickle

# Set location to store encodings
DEFAULT_ENCODEINGS_PATH = Path("out/encodings.pkl")

# Procedure to encode training data (encoding allows the model to interpret the data)
# By default, uses HOG (Historgram of Oriented Gradients)
def encode_known_faces(model: str = "hog", encodings_loc: Path = DEFAULT_ENCODEINGS_PATH) -> None:
    names = []
    encodings = []

    # Iterate over training data
    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model=model)  # Gets a tuple of 4 points for each face which indicates its location
        face_encodings = face_recognition.face_encodings(image, face_locations)  # Encodes each of the faces based on the locations provided

        for encoding in face_encodings:
            names.append(name)  # Add the data's face to the list
            encodings.append(encoding)  # Add the data's encoding to the list

    # Save the encodings to a dict which can be saved locally
    encodings_dict = {"names": names, "encodings": encodings}

    # Save the encodings locally
    with encodings_loc.open(mode="wb") as f:
        pickle.dump(encodings_loc, f)

# Call the procedure
encode_known_faces()