import numpy as np
from typing import Dict

# -----------------------------
# Credit Score Configuration
# -----------------------------

SCORE_MIN = 300
SCORE_MAX = 850

# Scorecard parameters
BASE_SCORE = 600     # Score at reference odds
BASE_ODDS = 1 / 19   # 5% default probability
PDO = 50             # Points to Double the Odds

FACTOR = PDO / np.log(2)
OFFSET = BASE_SCORE - FACTOR * np.log(BASE_ODDS)

def probability_to_score(prob: float) -> int:
    """
    Convert default probability to credit score.
    Monotonic, explainable, industry standard.
    """
    prob = np.clip(prob, 1e-6, 1 - 1e-6)

    odds = (1 - prob) / prob
    score = OFFSET + FACTOR * np.log(odds)

    score = int(round(score))
    score = max(SCORE_MIN, min(SCORE_MAX, score))

    return score

def assign_risk_band(score: int) -> str:
    if score >= 750:
        return "Very Low Risk"
    elif score >= 650:
        return "Low Risk"
    elif score >= 550:
        return "Medium Risk"
    elif score >= 450:
        return "High Risk"
    else:
        return "Very High Risk"

def generate_credit_score(probability: float) -> Dict:
    score = probability_to_score(probability)
    band = assign_risk_band(score)

    return {
        "risk_probability": round(probability, 4),
        "credit_score": score,
        "risk_band": band
    }
