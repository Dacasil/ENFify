import pickle
import warnings

import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch import nn
from tqdm import tqdm

# TODO: resolve warning
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.base")

SPATIAL_INPUT_SIZE = 46  # Height and width for CNN input
TEMPORAL_INPUT_SIZE = 25  # Input size for LSTM
SEQUENCE_LENGTH = 85  # Temporal sequence length


class CNNSpatialExtractor(nn.Module):
    def __init__(self, input_size):
        super(CNNSpatialExtractor, self).__init__()
        # Convolution layers with padding to preserve spatial dimensions
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        final_size = input_size // 8

        # Compute the flattened size after convolutions and pooling
        self.fc1 = nn.Linear(
            64 * final_size * final_size, 1024
        )  # Flattened size: 64 channels * 5x5
        self.fc2 = nn.Linear(1024, 256)  # Second fully connected layer

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Output size: (22x22)
        x = self.pool(F.relu(self.conv2(x)))  # Output size: (11x11)
        x = self.pool(F.relu(self.conv3(x)))  # Output size: (5x5)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = F.relu(self.fc1(x))  # First fully connected layer
        x = F.relu(self.fc2(x))  # Second fully connected layer
        return x


# BiLSTM block for temporal feature extraction
class DeepBiLSTMTemporalExtractor(nn.Module):
    def __init__(self, input_size=85, hidden_size=85, num_layers=2):
        super(DeepBiLSTMTemporalExtractor, self).__init__()

        # First BiLSTM module
        self.bilstm1 = nn.LSTM(
            input_size, hidden_size, num_layers=1, bidirectional=True, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_size * 2)  # Normalization layer for the first BiLSTM

        # Second BiLSTM module
        self.bilstm2 = nn.LSTM(
            hidden_size * 2, hidden_size, num_layers=1, bidirectional=True, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_size * 2)  # Normalization layer for the second BiLSTM

        # Fully connected layers
        self.fc1 = nn.Linear(
            hidden_size * 2, 512
        )  # First fully connected layer (input: hidden_size*2, output: 512)
        self.fc2 = nn.Linear(512, 256)  # Second fully connected layer (input: 512, output: 256)

    def forward(self, x):
        # First BiLSTM layer
        lstm_out1, _ = self.bilstm1(x)
        lstm_out1 = self.norm1(lstm_out1)  # Apply normalization
        lstm_out1 = F.relu(lstm_out1)  # Apply ReLU activation

        # Second BiLSTM layer
        lstm_out2, _ = self.bilstm2(lstm_out1)
        lstm_out2 = self.norm2(lstm_out2)  # Apply normalization
        lstm_out2 = F.relu(lstm_out2)  # Apply ReLU activation

        # Get the last time step output from the sequence
        x = lstm_out2[:, -1, :]  # Shape (batch_size, hidden_size * 2)

        # Fully connected layers
        x = F.relu(self.fc1(x))  # First fully connected layer
        x = F.relu(self.fc2(x))  # Second fully connected layer

        return x


class SpatioTemporalAttention(nn.Module):
    def __init__(self, spatial_feature_size, temporal_feature_size):
        super(SpatioTemporalAttention, self).__init__()

        self.concat_size = (
            spatial_feature_size + temporal_feature_size
        )  # Size of concatenated features

        # Fully connected layers for feature compression and transformation
        self.fc1 = nn.Linear(self.concat_size, self.concat_size)
        self.fc2 = nn.Linear(self.concat_size, self.concat_size // 8)
        self.fc3 = nn.Linear(self.concat_size // 8, self.concat_size)

        # Fully connected layer for attention weights
        self.fc4 = nn.Linear(self.concat_size, self.concat_size)

    def forward(self, spatial_feat, temporal_feat):
        # Concatenate spatial and temporal features
        combined_feat = torch.cat(
            (spatial_feat, temporal_feat), dim=1
        )  # Shape: (batch_size, concat_size)

        # Apply compression layers with ReLU activation
        x = F.relu(self.fc1(combined_feat))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # Compute attention weights using sigmoid activation
        attention_weights = torch.sigmoid(self.fc4(x))  # Shape: (batch_size, concat_size)

        # Element-wise multiplication to fuse features with attention weights
        fused_features = combined_feat * attention_weights  # Shape: (batch_size, concat_size)

        return fused_features


class ClassificationNetwork(nn.Module):
    def __init__(self, input_size):
        super(ClassificationNetwork, self).__init__()

        # Define fully connected layers
        self.fc1 = nn.Linear(input_size, 400)
        self.fc2 = nn.Linear(400, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 32)

        # Define dropout layer
        self.dropout = nn.Dropout(0.2)

        # Define final fully connected layer for output
        self.fc5 = nn.Linear(32, 1)  # Output is binary classification, so 2 neurons

    def forward(self, x):
        # Pass through the first fully connected layer and apply Leaky ReLU
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)

        # Pass through the second fully connected layer and apply Leaky ReLU
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout(x)

        # Pass through the third fully connected layer and apply Leaky ReLU
        x = F.leaky_relu(self.fc3(x))
        x = self.dropout(x)

        # Pass through the fourth fully connected layer and apply Leaky ReLU
        x = F.leaky_relu(self.fc4(x))
        x = self.dropout(x)

        # Output layer with softmax activation
        x = self.fc5(x)
        # output = F.log_softmax(x, dim=1)  # Use log_softmax for numerical stability

        return x


# Complete Network
class ParallelCNNBiLSTM(nn.Module):
    def __init__(self, temporal_input_size, spatial_input_size):
        super(ParallelCNNBiLSTM, self).__init__()
        self.spatial_extractor = CNNSpatialExtractor(input_size=spatial_input_size)
        self.temporal_extractor = DeepBiLSTMTemporalExtractor(
            input_size=temporal_input_size, hidden_size=temporal_input_size
        )
        self.attention = SpatioTemporalAttention(256, 256)
        self.classifier = ClassificationNetwork(input_size=2 * 256)

    def forward(self, spatial_input, temporal_input):
        spatial_features = self.spatial_extractor(spatial_input)
        temporal_features = self.temporal_extractor(temporal_input)
        fused_features = self.attention(spatial_features, temporal_features)
        output = self.classifier(fused_features)
        return output


def apply_normalization(features, scaler):
    init_shape = features.shape
    features = features.reshape(init_shape[0], -1)
    features = scaler.transform(features)
    features = features.reshape(init_shape)
    return features


def bilstm_classifier(
    model_path, spatial_scaler_path, temporal_scaler_path, spatial_features, temporal_features
):
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

    prediction = torch.sigmoid(output)
    prediction = (prediction > 0.5).int().item()
    return prediction


if __name__ == "__main__":
    import numpy as np

    model_path = "/home/cloud/enfify/models/cnn_bilstm_alldata_model.pth"
    spatial_scaler_path = "/home/cloud/enfify/models/cnn_bilstm_alldata_spatial_scaler.pkl"
    temporal_scaler_path = "/home/cloud/enfify/models/cnn_bilstm_alldata_temporal_scaler.pkl"

    spatial_features = np.load(
        "/home/cloud/enfify/data/test_bilstm/WHU_ref/whu_ref_spatial_freqs.npy", allow_pickle=True
    )
    temporal_features = np.load(
        "/home/cloud/enfify/data/test_bilstm/WHU_ref/whu_ref_temporal_freqs.npy", allow_pickle=True
    )
    labels = np.load(
        "/home/cloud/enfify/data/test_bilstm/WHU_ref/whu_ref_labels_freqs.npy", allow_pickle=True
    ).astype(int)

    predictions = []
    for spatial, temporal, label in zip(spatial_features, temporal_features, tqdm(labels)):
        prediction = bilstm_classifier(
            model_path, spatial_scaler_path, temporal_scaler_path, spatial, temporal
        )
        # print(f"{label}-{prediction}")
        predictions.append(prediction)

    cm = confusion_matrix(labels, predictions)
    print(cm)
