import numpy as np
import os
from glob import glob
import yaml
from scipy.io import wavfile
from tqdm import tqdm
from pathlib import Path
from enfify.config import ENFIFY_DIR
from enfify.pipeline import phase_CNNBiLSTM_feature_pipeline

# RAW_DATA_DIR = Path("/home/leo_dacasi/Dokumente/summerofcode/Enfify Data Synced/external/Carioca")
PROCESSED_DATA_DIR = Path("/home/leo_dacasi/Dokumente/summerofcode/Enfify Data Synced/processed")

# Paths
FEATURE_PATH = Path(PROCESSED_DATA_DIR / "Carioca1_original_phases_CNNBiLSTM")
os.makedirs(FEATURE_PATH, exist_ok=True)

"""
# Get Cut and Uncut Files
filenames = sorted(glob(str(Path(RAW_DATA_DIR / "Carioca1" / "*.wav"))))
cut_files = [f for f in filenames if "tamp" in f]
uncut_files = [f for f in filenames if "auth" in f]
"""

CARIOCA1_MEN = Path(
    "/home/leo_dacasi/Dokumente/summerofcode/Enfify Data Synced/external/Carioca/BASE CARIOCA 1/Homens"
)
CARIOCA1_WOMEN = Path(
    "/home/leo_dacasi/Dokumente/summerofcode/Enfify Data Synced/external/Carioca/BASE CARIOCA 1/Mulheres"
)
CARIOCA2_MEN_MOBILE = Path(
    "/home/leo_dacasi/Dokumente/summerofcode/Enfify Data Synced/external/Carioca/BASE CARIOCA 2/Movel/Homens"
)
CARIOCA2_MEN_LANDLINE = Path(
    "/home/leo_dacasi/Dokumente/summerofcode/Enfify Data Synced/external/Carioca/BASE CARIOCA 2/Fixo/Homens"
)
CARIOCA2_WOMEN_MOBILE = Path(
    "/home/leo_dacasi/Dokumente/summerofcode/Enfify Data Synced/external/Carioca/BASE CARIOCA 2/Movel/Mulheres"
)
CARIOCA2_WOMEN_LANDLINE = Path(
    "/home/leo_dacasi/Dokumente/summerofcode/Enfify Data Synced/external/Carioca/BASE CARIOCA 2/Fixo/Mulheres"
)

cut_files_car = []
uncut_files_car = []


car1_men = sorted(glob(str(Path(CARIOCA1_MEN / "*.wav"))))
for file in car1_men:
    file_name = Path(file).name
    if "e_sem_60" not in file_name and "e" not in file_name:
        uncut_files_car.append(file)
        print(f"UNCUT: {file_name}")
    elif "_sem_60" not in file_name and "e" in file_name:
        cut_files_car.append(file)
        print(f"CUT: {file_name}")

print("LEERZEICHEN")

car1_women = sorted(glob(str(Path(CARIOCA1_WOMEN / "*.wav"))))
for file in car1_women:
    file_name = Path(file).name
    if "e_sem_60" not in file_name and "e" not in file_name:
        uncut_files_car.append(file)
        print(f"UNCUT: {file_name}")
    elif "_sem_60" not in file_name and "e" in file_name:
        cut_files_car.append(file)
        print(f"CUT: {file_name}")

print("LEERZEICHEN")

car2_men_land = sorted(glob(str(Path(CARIOCA2_MEN_LANDLINE / "*.wav"))))
for file in car2_men_land:
    file_name = Path(file).name
    if "e" not in file_name:
        uncut_files_car.append(file)
        print(f"UNCUT: {file_name}")
    elif "e" in file_name:
        cut_files_car.append(file)
        print(f"CUT: {file_name}")

print("LEERZEICHEN")

car2_men_mobile = sorted(glob(str(Path(CARIOCA2_MEN_MOBILE / "*.wav"))))
for file in car2_men_mobile:
    file_name = Path(file).name
    if "e" not in file_name:
        uncut_files_car.append(file)
        print(f"UNCUT: {file_name}")
    elif "e" in file_name:
        cut_files_car.append(file)
        print(f"CUT: {file_name}")

print("LEERZEICHEN")

car2_women_land = sorted(glob(str(Path(CARIOCA2_WOMEN_LANDLINE / "*.wav"))))
for file in car2_women_land:
    file_name = Path(file).name
    if "e" not in file_name:
        uncut_files_car.append(file)
        print(f"UNCUT: {file_name}")
    elif "e" in file_name:
        cut_files_car.append(file)
        print(f"CUT: {file_name}")

print("LEERZEICHEN")

car2_women_mobile = sorted(glob(str(Path(CARIOCA2_WOMEN_MOBILE / "*.wav"))))
for file in car2_women_mobile:
    file_name = Path(file).name
    if "e" not in file_name:
        uncut_files_car.append(file)
        print(f"UNCUT: {file_name}")
    elif "e" in file_name:
        cut_files_car.append(file)
        print(f"CUT: {file_name}")


uncut_spatial = []
uncut_temporal = []
cut_spatial = []
cut_temporal = []

for file in tqdm(cut_files_car):
    basename = os.path.splitext(os.path.basename(file))[0]
    sample_freq, sig = wavfile.read(file)
    with open(ENFIFY_DIR / "config_springer.yml", "r") as f:
        config = yaml.safe_load(f)
    cut_spatial_phase, cut_temporal_phase = phase_CNNBiLSTM_feature_pipeline(
        sig, sample_freq, config
    )
    print(file)
    cut_spatial.append(cut_spatial_phase)
    cut_temporal.append(cut_temporal_phase)

for file in tqdm(uncut_files_car):
    basename = os.path.splitext(os.path.basename(file))[0]
    sample_freq, sig = wavfile.read(file)
    with open(ENFIFY_DIR / "config_springer.yml", "r") as f:
        config = yaml.safe_load(f)
    uncut_spatial_phase, uncut_temporal_phase = phase_CNNBiLSTM_feature_pipeline(
        sig, sample_freq, config
    )
    print(file)
    uncut_spatial.append(uncut_spatial_phase)
    uncut_temporal.append(uncut_temporal_phase)


# Label Combine and shuffle the Data
labels = np.concatenate([np.ones(len(cut_spatial)), np.zeros(len(cut_temporal))])
spatial_data = uncut_spatial + cut_spatial
temporal_data = uncut_temporal + cut_temporal
indices = np.random.permutation(len(spatial_data))

print(f"Spatial Data {len(spatial_data)}")
print(f"Temporal Data {len(temporal_data)}")

spatial_data = np.array([spatial_data[i] for i in indices], dtype=object)
temporal_data = np.array([temporal_data[i] for i in indices], dtype=object)
labels = labels[indices]

# Save the data
np.save(str(FEATURE_PATH) + "/" + "carioca1_temporal_phase_10s.npy", temporal_data)
np.save(str(FEATURE_PATH) + "/" + "carioca1_spatial_phase_10s.npy", spatial_data)
np.save(str(FEATURE_PATH) + "/" + "carioca1_labels_10s_phase.npy", labels)
