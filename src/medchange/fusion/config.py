from __future__ import annotations


BEST_BIOMEDCLIP_FEATURES = {
    "atelectasis":
        "prior_current_delta",

    "cardiomegaly":
        "full",

    "consolidation":
        "full",

    "edema":
        "prior_current",

    "pleural_effusion":
        "prior_current_delta",

    "pneumonia":
        "prior_current_delta",

    "pneumothorax":
        "full",
}


FUSION_FINDINGS = list(
    BEST_BIOMEDCLIP_FEATURES.keys()
)