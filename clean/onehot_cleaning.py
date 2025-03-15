import numpy as np


def process_multihot(input_string: str, elements: list[str]):
    """
    Converts a comma-separated string into a multi-hot encoded vector and
    then expands it into a matrix of one-hot vectors.
    >>> process_multihot("apple,banana", ["apple", "orange", "banana", "grape"])
    array([[1, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 1, 0],
           [0, 0, 0, 0]])
    """
    input_list = list(item.strip() for item in input_string.split(","))
    multihot = np.isin(elements, input_list).astype(int)
    onehot = np.diag(multihot)
    return onehot


def process_onehot(input_string: str, elements: list[str]):
    """
    Converst a single label into a one-hot vector.
    >>> process_onehot("apple", ["apple", "orange", "banana"])
    array([1, 0, 0])
    """
    return np.array([1 if item == input_string else 0 for item in elements])


if __name__ == "__main__":
    import doctest
    doctest.testmod()
