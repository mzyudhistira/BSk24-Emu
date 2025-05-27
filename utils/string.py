import random


def generate_random_hex(k=5):
    """
    Generate a random hexadecimal based string

    Args:
        k (int) : The length of the string, defaults to 5

    return:
        hex_string (str): A k-digit hexadecimal-based string

    """
    hex_chars = "0123456789ABCDEF"
    hex_string = "".join(random.choices(hex_chars, k=k))

    return hex_string
