import re
from datetime import datetime
from typing import Optional

NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90
}


def _spelled_to_int(spelled: str) -> Optional[int]:
    """
    Parses spelled out numbers like fifty-six or fifty six
    """
    parts = re.split(r"[\s-]+", spelled.lower().strip())
    total = 0
    for p in parts:
        if p not in NUM_WORDS:
            return None
        total += NUM_WORDS[p]
    return total


def _parse_single_number(token: str) -> Optional[int]:
    """Parse a single token as an integer (numeric or spelled-out)."""
    if re.fullmatch(r"\d+", token):
        return int(token)
    return _spelled_to_int(token)


def _parse_range(token: str) -> Optional[list[int]]:
    """Detect and extract numeric ranges ('5-10' or 'five-ten')"""
    if "-" not in token:
        return None
    if _parse_single_number(token) is not None:
        return None
    parts = token.split("-", 1)
    if len(parts) != 2:
        return None
    left_val, right_val = _parse_single_number(parts[0]), _parse_single_number(parts[1])
    return [left_val, right_val] if left_val is not None and right_val is not None else None


def process_string(text: str) -> float:
    """
    1) Check if we have a date and return the modified version.
    2) Remove all dollar signs from the text.
    3) Convert ranges like '5 - 10' into '5-10' by removing spaces around the dash.
    4) Extract all numbers (digits or spelled-out) or ranges; for each range, interpret both endpoints.
    5) If any numbers are found, return the average; otherwise return the word count.

    >>> process_string("6/7/2025") == 6.5
    True
    >>> process_string("5-Jun") == 5.5
    True
    >>> process_string("I have fifty six apples and 3 bananas") == 29.5
    True
    >>> process_string("Ranges like 5-10 or 5 - 10 should be parsed as 5 and 10") == 7.5
    True
    >>> process_string("No numbers here at all") == 5
    True
    >>> process_string("fifty-six is a single number, not a range") == 56
    True
    >>> process_string("I might say twenty 31 or 100-200 or $5-$10") == 61
    True
    >>> process_string("ninety nine bottles of beer") == 99
    True
    >>> process_string("thirty five - forty five should be 35 and 45") == 40
    True
    """

    date_pattern = r"^(\d{1,2})/(\d{1,2})/\d{4}$"
    match = re.match(date_pattern, text)
    if match:
        month, day = match.groups()
        return (int(month) + int(day)) / 2

    try:
        parsed_date = datetime.strptime(text, "%d-%b")
        return (parsed_date.month + parsed_date.day) / 2
    except ValueError:
        pass

    text = text.replace("$", "")
    # Remove white space 5 - 10 -> 5-10
    text = re.sub(r"\s*-\s*", "-", text)

    words = text.split()
    tokens = []
    skip = False

    for i in range(len(words)):
        if skip:
            skip = False
            continue

        # Try to form two-word numbers like "fifty six"
        if i < len(words) - 1:
            combined = words[i] + " " + words[i + 1]
            if _spelled_to_int(combined) is not None:
                tokens.append(combined)
                skip = True
                continue

        tokens.append(words[i])

    numbers_found = []

    for token in tokens:
        rng = _parse_range(token)
        if rng is not None:
            numbers_found.extend(rng)
        else:
            val = _parse_single_number(token)
            if val is not None:
                numbers_found.append(val)

    return sum(numbers_found) / len(numbers_found) if numbers_found else len(words)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
