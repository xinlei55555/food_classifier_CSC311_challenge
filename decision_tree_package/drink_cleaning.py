from collections import OrderedDict
import re


def parse_common_drinks(input_simple_file: str = "common_drinks.simple") -> OrderedDict[str, str]:
    """
    Parses the common_drinks.simple file and returns an ordered dict of mappings from synonyms
    to their respective drinks (so reverse the common_drinks.simple relation)
    """
    drink_dict = OrderedDict()
    with open(input_simple_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.split("#", 1)[0].strip() == "":
                continue
            drink, synonyms_str = line.split("=", 1)
            synonyms = [syn.strip() for syn in synonyms_str.split(",")]
            for synonym in synonyms:
                drink_dict[synonym] = drink
    return drink_dict


def parse_drinks_list(input_simple_file: str = "common_drinks.simple") -> OrderedDict[str, str]:
    """
    Parses the common_drinks.simple file and returns a list of all drinks in the file.
    """
    drink_list = []
    with open(input_simple_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.split("#", 1)[0].strip() == "":
                continue
            drink, _ = line.split("=", 1)
            drink_list.append(drink)
    return drink_list

def process_drink(input_str: str, common_drinks: OrderedDict, default: str = "none") -> str:
    """
    1. Replace all forward slashes with a space.
    2. Remove all special non-alphabetic characters (including digits) except dashes.
    3. Lowercase the string.
    4. Reduce consecutive whitespace to a single space.
    5. Check if the entire normalized string matches any key in the ordered dictionary;
       if found, return that dictionary value.
    6. If no direct match, split on whitespace; for each word, see if there's a match.
       If found, return the dictionary value of that match.
    7. If still no match, compute a simple similarity score between each word in the
       splitted string and every dictionary key; pick the best match. Print a warning
       and return that best match's dictionary value.
    """

    text = input_str.replace("/", " ")
    text = re.sub(r"[^a-zA-Z\- ]+", "", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    # Direct match
    for k in common_drinks:
        if k == text:
            return common_drinks[k]
    # split on word
    words = text.split()
    for w in words:
        for k in common_drinks:
            if k == w:
                return common_drinks[k]

    # If still no match, compute a basic similarity score
    # Let's define a simple character-based Jaccard similarity:
    #   similarity = |set(a) ∩ set(b)| / |set(a) ∪ set(b)|
    # We'll search for the word/key pair that yields the highest similarity
    def jaccard_similarity(a: str, b: str) -> float:
        set_a = set(a)
        set_b = set(b)
        intersection = set_a.intersection(set_b)
        union = set_a.union(set_b)
        return len(intersection) / len(union) if union else 0.0

    best_score = -1.0
    best_key = None
    # For each word in the splitted string, check all dictionary keys
    for w in words:
        for k in common_drinks:
            score = jaccard_similarity(w, k)
            if score > best_score:
                best_score = score
                best_key = k

    # If we found something with best similarity, return that
    if best_key is not None and best_score > 0.5:
        # print(f"WARNING: failed to find good match for '{text}', returning '{best_key}' instead by similarity")
        return common_drinks[best_key]

    # If there's absolutely nothing (e.g., empty dictionary?), return some fallback
    # or just the original text
    # print(f"WARNING: no possible matches at all for '{text}', returning default '{default}'")
    return default

# -------------------------------------------------------------------
# Example Usage
if __name__ == "__main__":
    # Suppose we have the following OrderedDict:
    from collections import OrderedDict
    common_drinks = OrderedDict([
        ("boba", "boba"),
        ("bubble", "boba"),
        ("pearl", "boba"),
        ("bubble-tea", "boba"),
        ("milk shake", "milk-shake"),
        ("milk-shake", "milk-shake"),
    ])

    tests = [
        "Bubble-tea!!!",
        "milk Shake  /  ??? ",
        "milk-shake",
        "pearl tea",
        "milk shakeeee"
    ]

    for t in tests:
        result = process_drink(t, common_drinks)
        print(f"Input: {t!r} -> Result: {result}")
