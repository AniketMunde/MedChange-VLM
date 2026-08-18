from __future__ import annotations


def dashboard_css() -> str:
    return """
    <style>

    .block-container {
        max-width: 1450px;
        padding-top: 1.7rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid #E5E7EB;
    }

    h1 {
        letter-spacing: -0.03em;
    }

    h2, h3 {
        letter-spacing: -0.015em;
    }

    .medchange-header {
        padding: 0.2rem 0 1.2rem 0;
        border-bottom: 1px solid #E5E7EB;
        margin-bottom: 1.4rem;
    }

    .medchange-title {
        font-size: 2.05rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }

    .medchange-subtitle {
        color: #6B7280;
        font-size: 0.97rem;
        margin-bottom: 0;
    }

    .section-card {
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 1.1rem 1.2rem;
        background: rgba(255, 255, 255, 0.015);
        margin-bottom: 1rem;
    }

    .metric-card {
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        min-height: 112px;
    }

    .metric-label {
        color: #6B7280;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.4rem;
    }

    .metric-value {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.15rem;
    }

    .metric-caption {
        color: #6B7280;
        font-size: 0.79rem;
    }

    .status-low {
        font-weight: 650;
    }

    .status-moderate {
        font-weight: 650;
    }

    .status-high {
        font-weight: 700;
    }

    .finding-title {
        font-weight: 650;
    }

    .report-box {
        border-left: 4px solid #9CA3AF;
        background: rgba(127, 127, 127, 0.06);
        border-radius: 8px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
    }

    .review-box {
        border: 1px solid #D1D5DB;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.65rem;
    }

    .model-chip {
        display: inline-block;
        border: 1px solid #D1D5DB;
        border-radius: 999px;
        padding: 0.18rem 0.55rem;
        font-size: 0.72rem;
        margin-right: 0.35rem;
    }

    div[data-testid="stFileUploader"] {
        border-radius: 12px;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        overflow: hidden;
    }

    .footer-note {
        color: #6B7280;
        font-size: 0.77rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #E5E7EB;
    }

    </style>
    """