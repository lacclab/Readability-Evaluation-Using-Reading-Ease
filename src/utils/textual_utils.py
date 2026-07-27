
import textstat

def smog_index_fixed(text: str) -> float:
    r"""Calculate the SMOG index.

    Parameters
    ----------
    text : str
        A text string.

    Returns
    -------
    float
        The SMOG index for `text`.

    Notes
    -----
    The SMOG index is calculated as:

    .. math::

        (1.043*(30*(n\ polysyllabic\ words/n\ sentences))^{.5})+3.1291

    Polysyllabic words are defined as words with more than 3 syllables.
    """
    sentences = textstat.sentence_count(text)
    poly_syllab = textstat.polysyllabcount(text)
    try:
        smog = (
                (1.043 * (30 * (poly_syllab / sentences)) ** .5)
                + 3.1291)
        return textstat.textstat._legacy_round(smog, 1)
    except ZeroDivisionError:
        return 0.0
    

def clean_text(raw_text: str) -> str:
    """
    Replaces the problematic characters in the raw_text, made for OnestopQA.
    E.g., "ë" -> "e"
    """
    if type(raw_text) != str:
        return None

    return (
        raw_text.replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("…", "...")
        .replace("‘", "'")
        .replace("é", "e")
        .replace("ë", "e")
        .replace("ﬁ", "fi")
        .replace("ï", "i")
        .replace("‚", ",")
        .replace(" ", " ")
    )


def count_sentences(paragraph):
    """
    Count the number of sentences in a given paragraph.

    Args:
        paragraph (str): The text to analyze

    Returns:
        int: Number of sentences found

    Note: Considers '.', '!', and '?' as sentence endings.
    """
    if not isinstance(paragraph, str):
        raise TypeError("Input must be a string")

    if not paragraph.strip():
        return 0

    # Split on common sentence endings
    sentence_endings = [
        ". ",
        "! ",
        "? ",
        ".\n",
        "!\n",
        "?\n",
        ".\" ",
        "!\" ",
        "?\" ",
        ".\"\n",
        "!\"\n",
        "?\"\n",
    ]
    count = 0

    # Handle the case where the paragraph ends without space
    if paragraph.strip()[-1] in ".!?\"":
        count = 1

    for ending in sentence_endings:
        count += paragraph.count(ending)

    return count


def split_into_sentences(paragraph):
    """
    Split a given paragraph into sentences.

    Args:
        paragraph (str): The text to split.

    Returns:
        list: A list of sentences.

    Note: Considers '.', '!', and '?' as sentence endings, including cases with quotes.
    """
    if not isinstance(paragraph, str):
        raise TypeError("Input must be a string")

    if not paragraph.strip():
        return []

    # Sentence-ending markers
    sentence_endings = [
        ". ",
        "! ",
        "? ",
        ".\n",
        "!\n",
        "?\n",
        ".\" ",
        "!\" ",
        "?\" ",
        ".\"\n",
        "!\"\n",
        "?\"\n",
    ]

    sentences = []
    sentence = ""

    i = 0
    while i < len(paragraph):
        sentence += paragraph[i]

        # Check if the current position is the end of a sentence
        for ending in sentence_endings:
            if paragraph[i:].startswith(ending):
                sentences.append(sentence.strip())  # Store the complete sentence
                sentence = ""  # Reset for the next sentence
                i += len(ending) - 1  # Skip past the sentence ending
                break  # Stop checking endings for this character

        i += 1

    # Add any remaining text as a final sentence
    if sentence.strip():
        sentences.append(sentence.strip())

    return sentences