import pickle
import warnings

import torch

from enfify.networks import ParallelCNNBiLSTM
from enfify.preprocessing import extract_spatial_features, extract_temporal_features

# TODO: resolve warning
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.base")

SPATIAL_INPUT_SIZE = 46  # Height and width for CNN input
TEMPORAL_INPUT_SIZE = 25  # Input size for LSTM
SEQUENCE_LENGTH = 85  # Temporal sequence length


def apply_normalization(features, scaler):
    init_shape = features.shape
    features = features.reshape(init_shape[0], -1)
    features = scaler.transform(features)
    features = features.reshape(init_shape)
    return features


def bilstm_classifier(feature_freq, config, model_path, spatial_scaler_path, temporal_scaler_path):
    spatial_features = extract_spatial_features(feature_freq, config["bilstm_sn"])
    temporal_features = extract_temporal_features(
        feature_freq, config["bilstm_fl"], config["bilstm_fn"]
    )

    # Add dimension
    spatial_features = spatial_features[None, ...]
    temporal_features = temporal_features[None, ...]

    # Normalize
    with open(spatial_scaler_path, "rb") as f:
        scaler_spatial = pickle.load(f)
    spatial_features = apply_normalization(spatial_features, scaler_spatial)
    with open(temporal_scaler_path, "rb") as f:
        scaler_temporal = pickle.load(f)
    temporal_features = apply_normalization(temporal_features, scaler_temporal)

    spatial_features = torch.tensor(spatial_features, dtype=torch.float32)
    temporal_features = torch.tensor(temporal_features, dtype=torch.float32)

    spatial_features = spatial_features.unsqueeze(0)
    # temporal_features = temporal_features

    model = ParallelCNNBiLSTM(
        temporal_input_size=TEMPORAL_INPUT_SIZE, spatial_input_size=SPATIAL_INPUT_SIZE
    )

    model_state_dict = torch.load(model_path, weights_only=True, map_location=torch.device("cpu"))
    model.load_state_dict(model_state_dict)

    model.eval()
    with torch.no_grad():
        output = model(spatial_features, temporal_features)

    score = torch.sigmoid(output).item()  # Value between 0 and 1
    prediction = int((score > 0.5))  # 1 for tampered, 0 for authentic
    confidence = score if prediction else 1 - score

    return prediction, confidence


# TODO: remove when not needed
if __name__ == "__main__":
    from pathlib import Path

    import librosa
    from sklearn.metrics import confusion_matrix, accuracy_score
    from tqdm import tqdm

    from enfify.config import DEFAULT_CONFIG
    from enfify.pipeline import feature_freq_pipeline

    model_path = "/home/cloud/enfify/models/cnn_bilstm_alldata_model.pth"
    spatial_scaler_path = "/home/cloud/enfify/models/cnn_bilstm_alldata_spatial_scaler.pkl"
    temporal_scaler_path = "/home/cloud/enfify/models/cnn_bilstm_alldata_temporal_scaler.pkl"

    # Load data
    files = sorted(Path("/home/cloud/enfify/data/interim/WHU_ref").glob("*.wav"))[:100]

    predictions = []
    labels = []
    for file in (pbar := tqdm(files)):
        label = int("tamp" in file.name)
        labels.append(label)

        sig, sample_freq = librosa.load(file)
        feature_freq = feature_freq_pipeline(sig, sample_freq, DEFAULT_CONFIG)
        prediction, confidence = bilstm_classifier(
            feature_freq, DEFAULT_CONFIG, model_path, spatial_scaler_path, temporal_scaler_path
        )
        predictions.append(prediction)

        pbar.set_postfix_str(
            f"Label: {label}, Prediction: {prediction}, Confidence: {confidence:.2f}"
        )

    cm = confusion_matrix(labels, predictions)
    print(f"Confusion Matrix:\n{cm}")
    accuracy = accuracy_score(labels, predictions)
    print(f"Accuracy: {accuracy}")
