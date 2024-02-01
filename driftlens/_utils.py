from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).parent.parent

def clear_complex_numbers(scores):
    return [complex(score).real if type(score) == str else score for score in scores]

