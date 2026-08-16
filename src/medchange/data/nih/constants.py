from __future__ import annotations


HF_NIH_DATASET = (
    "BahaaEldin0/NIH-Chest-Xray-14"
)


NIH_TO_MEDCHANGE = {
    "Atelectasis": "atelectasis",
    "Cardiomegaly": "cardiomegaly",
    "Consolidation": "consolidation",
    "Edema": "edema",
    "Effusion": "pleural_effusion",
    "Pneumonia": "pneumonia",
    "Pneumothorax": "pneumothorax",
}


MEDCHANGE_TO_NIH = {
    value: key
    for key, value
    in NIH_TO_MEDCHANGE.items()
}


TARGET_FINDINGS = list(
    MEDCHANGE_TO_NIH.keys()
)


SUPPORTED_VIEWS = {
    "AP",
    "PA",
}