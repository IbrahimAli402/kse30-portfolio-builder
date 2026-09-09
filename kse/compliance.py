"""
kse/compliance.py
==================
Compliance layer for the KSE 100 Portfolio Builder.

Provides:
    - get_disclaimer()         — short and full disclaimer text
    - get_data_provenance()    — data source metadata for transparency
    - assess_suitability()     — map questionnaire answers to risk tier
    - get_suitability_questions() — load questions from config
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional


_CONFIG_DIR = Path(__file__).parent.parent / "config"


def _load_config() -> Dict:
    """Load disclaimers and compliance config."""
    path = _CONFIG_DIR / "disclaimers.yaml"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_disclaimer(kind: str = "short") -> str:
    """
    Get disclaimer text.

    Parameters
    ----------
    kind : str
        'short' for one-line, 'full' for complete disclaimer.

    Returns
    -------
    str
    """
    config = _load_config()
    return config.get("disclaimer", {}).get(kind, "Not investment advice.")


def get_data_provenance(source_key: Optional[str] = None) -> Dict:
    """
    Get data source metadata for transparency.

    Parameters
    ----------
    source_key : str, optional
        Specific source key (e.g., 'kse100_returns'). If None, returns all.

    Returns
    -------
    dict
    """
    config = _load_config()
    sources = config.get("data_sources", {})

    if source_key:
        return sources.get(source_key, {})
    return sources


def get_suitability_questions() -> List[Dict]:
    """
    Load suitability questionnaire questions from config.

    Returns
    -------
    list of dicts with id, question, options, scores
    """
    config = _load_config()
    return config.get("suitability", {}).get("questions", [])


def get_suitability_thresholds() -> Dict:
    """
    Get score thresholds for risk tier mapping.

    Returns
    -------
    dict with conservative, moderate, aggressive thresholds
    """
    config = _load_config()
    return config.get("suitability", {}).get("thresholds", {
        "conservative": 4,
        "moderate": 7,
        "aggressive": 10,
    })


def assess_suitability(answers: Dict[str, int]) -> Dict:
    """
    Map questionnaire answers to a recommended risk tier.

    Parameters
    ----------
    answers : dict
        Mapping of question_id → selected_option_index (0-based).

    Returns
    -------
    dict with:
        - total_score : int
        - recommended_tier : str ('Conservative', 'Moderate', 'Aggressive')
        - tier_reason : str — explanation of the recommendation
    """
    questions = get_suitability_questions()
    thresholds = get_suitability_thresholds()

    total_score = 0
    for q in questions:
        qid = q["id"]
        if qid in answers:
            idx = answers[qid]
            if 0 <= idx < len(q["scores"]):
                total_score += q["scores"][idx]

    if total_score <= thresholds["conservative"] + 2:
        tier = "Conservative"
        reason = "Your answers suggest a lower risk tolerance or shorter horizon. A conservative allocation (60% equity / 40% income) prioritizes capital preservation."
    elif total_score <= thresholds["moderate"] + 2:
        tier = "Moderate"
        reason = "Your answers suggest a balanced risk profile. A moderate allocation (80% equity / 20% income) balances growth potential with risk management."
    else:
        tier = "Aggressive"
        reason = "Your answers suggest a higher risk tolerance and longer horizon. An aggressive allocation (100% equity) maximizes growth potential but comes with higher volatility."

    return {
        "total_score": total_score,
        "recommended_tier": tier,
        "tier_reason": reason,
    }