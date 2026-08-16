EVALUATION_FINDINGS = [
    "atelectasis",
    "cardiomegaly",
    "consolidation",
    "edema",
    "pleural_effusion",
    "pneumonia",
    "pneumothorax",
]


MIMIC_TO_INTERNAL_LABEL = {
    "Atelectasis": "atelectasis",
    "Cardiomegaly": "cardiomegaly",
    "Consolidation": "consolidation",
    "Edema": "edema",
    "Pleural Effusion": (
        "pleural_effusion"
    ),
    "Pneumonia": "pneumonia",
    "Pneumothorax": "pneumothorax",
}