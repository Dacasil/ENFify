from math import ceil
from loguru import logger
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import robust_scale

from enfify.config import DATA_DIR
from enfify.networks import OneDCNN


def cnn_classifier(model_path, feature_vector):
    """Load a pre-trained CNN model and classify a feature vector.

    Args:
        model_path (str): Path to the saved model.
        feature_vector (numpy.ndarray): Input feature vector to classify.

    Returns:
        Tuple[int, float]: The predicted class (0 or 1) and the confidence for the prediction.
    """
    # Load the saved model
    model = OneDCNN()
    model.load_state_dict(
        torch.load(model_path, weights_only=True, map_location=torch.device("cpu"))
    )
    model.eval()  # Set model to evaluation mode

    # Normalize
    feature_vector = robust_scale(feature_vector[None, :], axis=1)
    feature_vector = feature_vector.squeeze(0)

    # Convert feature vector to a torch tensor and add batch and channel dimensions
    feature_vector = torch.tensor(feature_vector, dtype=torch.float32)
    feature_vector = feature_vector.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, feature_length)

    # Run the feature vector through the model
    with torch.no_grad():  # Disable gradient computation
        output = model(feature_vector)

    # Apply softmax to get probabilities
    probabilities = F.softmax(output, dim=1)

    # Get the predicted class and its confidence
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_class].item()  # Confidence for the predicted class

    return predicted_class, confidence


def sectioning(array, section_len, min_overlap):
    """Split an array into sections of a given length with a minimum overlap so that the whole array is covered.
    Used e.g. for mapping out a feature array for a CNN with a certain input length.

    Args:
        array (numpy.ndarray): Array to split.
        section_len (int): Length of each section.
        min_overlap (int): Minimum overlap between sections.

    Returns:
        List[numpy.ndarray]: List of sections.
    """
    start = np.linspace(
        0,
        len(array) - section_len,
        ceil(len(array) / (section_len - min_overlap)),
        dtype=int,
    )
    logger.debug(f"Start: {start}")
    end = start + section_len
    return [array[s:e] for s, e in zip(start, end)]


if __name__ == "__main__":
    model_path = "/home/cloud/enfify/models/onedcnn_model_carioca_83.pth"
    filepath = DATA_DIR / "processed" / "Carioca1" / "HC01-00-tamp.npy"
    filename = filepath.name
    feature_vec = np.load(filepath)
    feature_vec = feature_vec[40:-40]

    predicted_class, confidence = cnn_classifier(model_path, feature_vec)
    print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}")
