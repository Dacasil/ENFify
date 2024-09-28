from sklearn.preprocessing import RobustScaler


def normalize_robust(data):

    init_shape = data.shape
    # if len(data.shape) == 1:
    data = data.reshape(data.shape[0], -1)

    # Apply RobustScaler
    scaler_spatial = RobustScaler()
    data_normalized = scaler_spatial.fit_transform(data)

    # Reshape back to original shape
    data_normalized = data_normalized.reshape(init_shape)

    return data_normalized
