from pathlib import Path


def get_project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent


def clear_complex_numbers(scores):
    """Clears complex numbers from a list of scores (distances).
    Args:
        scores (list): List of scores (distances).
    Returns:
        list: List of scores (distances) with complex numbers cleared.
    """
    return [complex(score).real if type(score) == str else score for score in scores]

