"""
utils
"""


def load_txt(path: str) -> list:
    return [line.rstrip('\n') for line in open(path)]
