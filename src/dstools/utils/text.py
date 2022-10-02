from collections import Counter
from typing import Tuple


def get_yules_complexity_from_tokens(tokens: list[str]) -> Tuple[float, float]:
    """
    Takes a list of tokens and returns a tuple with two basic measures for lexical complexity: Yule's K and Yule's I.
    (cf. Oakes, M.P. 1998. Statistics for Corpus Linguistics.
    International Journal of Applied Linguistics, Vol 10 Issue 2)

    Args:
        tokens: list of text tokens

    Returns:
        tuple with Yule's K and Yule's I
    """
    token_counter = Counter(tok.upper() for tok in tokens)
    m1 = sum(token_counter.values())
    m2 = sum([freq ** 2 for freq in token_counter.values()])
    i = (m1 * m1) / (m2 - m1)
    k = 10000 / i
    return (k, i)


def time_str_to_seconds(time_str: str) -> int:
    """
    Converts a time string to seconds.

    Args:
        time_str: string in the format "HH:MM:SS" or  "MM:SS" or "SS"
    """
    parts = list(map(int, time_str.split(":")))
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:
        print(parts)
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 1:
        return int(parts[0])
    else:
        raise ValueError("Invalid time string, needs to be one of HH:MM:SS, MM:SS or SS")


def height_str_to_cm(height_str: str, digits: int = 1) -> float:
    """
    Converts a height string to centimeters.

    Args:
        height_str: string in the format FT-IN or FT'IN" or just FT' or just IN"
        digits: number of digits to round to
    Returns:
        height in centimeters
    """
    height_str = height_str.replace(" ", "")  # remove superfluous spaces
    if "-" in height_str:
        feet, inches = map(int, height_str.split("-"))
    elif "'" in height_str and height_str.endswith('"'):
        feet, inches = map(int, height_str[:-1].split("'"))  # removes final " before splitting on '
    elif height_str.rstrip().endswith("'"):
        feet = int(height_str[:-1])
        inches = 0
    elif height_str.rstrip().endswith('"') and "'" not in height_str:
        feet = 0
        inches = int(height_str[:-1])
    else:
        raise ValueError(f"Invalid height string, needs to be in the format FT-IN or FT'IN\". Received: {height_str}")
    return round(30.48 * feet + inches * 2.54, digits)
