"""
Code adapted from https://github.com/jkkummerfeld/text2sql-data/
"""

import nltk
import re


def process_sentence(sentence):
    sentence = dept_num_spacing(sentence)
    sentence = am_and_pm(sentence)
    sentence = standardize_word_forms(sentence)
    sentence = n_credit(sentence)
    return sentence


def n_credit(sentence):
    """
    If a phrase in the form "X credit" appears in the sentence, and
    X is a number, add a hyphen to "credit".
    If "X-credit" appears, split it into "X" and "-credit".
    Do not hyphenate "X credits" or "Y credit" where Y is not a number.
    Run this after the tokenized text has been joined.
    >>> n_credit('I need a 3 credit course .')
    'I need a 3 -credit course .'
    >>> n_credit('I need a 3-credit course .')
    'I need a 3 -credit course .'
    >>> n_credit('I need a course worth 3 credits .')
    'I need a course worth 3 credits .'
    >>> n_credit('Can I get credit ?')
    'Can I get credit ?'
    """
    pattern = r"(?P<number>\d)+[- ]credit\s"
    repl = r"\g<number> -credit "
    return re.sub(pattern, repl, sentence)


def dept_num_spacing(sentence):
    """
    Given a sentence with a department abbreviation followed by a course number,
    ensure that there's a space between the abbreviation and number.
    An all-caps string of exactly 4 letters or the string "eecs" is considered
    a department if it is followed immediately by a 3-digit number.
    Run this before tokenizing.
    >>> dept_num_spacing("EECS280")
    'EECS 280'
    >>> dept_num_spacing("MATH417")
    'MATH 417'
    >>> dept_num_spacing("eecs280")
    'eecs 280'
    >>> dept_num_spacing("gEECS365")
    'gEECS365'
    >>> dept_num_spacing("EECS280 and MATH417")
    'EECS 280 and MATH 417'
    """
    pattern = r"(?P<dept>^[A-Z]{4}|\s[A-Z]{4}|eecs)(?P<number>\d{3})"
    repl = r"\g<dept> \g<number>"
    return re.sub(pattern, repl, sentence)


def am_and_pm(sentence):
    """
    Standardize variations as "A.M." or "P.M." iff they appear after a time.
    >>> am_and_pm("at twelve pm")
    'at twelve P.M.'
    >>> am_and_pm("at 12 pm")
    'at 12 P.M.'
    >>> am_and_pm("I am on a boat")
    'I am on a boat'
    >>> am_and_pm("9 am classes")
    '9 A.M. classes'
    >>> am_and_pm("9 AM classes")
    '9 A.M. classes'
    >>> am_and_pm("9:30 AM classes")
    '9:30 A.M. classes'
    >>> am_and_pm("9AM classes")
    '9 A.M. classes'
    >>> am_and_pm("is 280 among")
    'is 280 among'
    """
    number_pattern = r"(?P<time>(^|\s|[A-Za-z:])\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|thirty|forty-?five|o'clock) ?(?P<meridian>"
    am_pattern = number_pattern + r"am|AM|a\.m\.|A\.M\.)"
    pm_pattern = number_pattern + r"pm|PM|p\.m\.|P\.M\.)"

    am_repl = r"\g<time> A.M."
    pm_repl = r"\g<time> P.M."

    sentence = re.sub(am_pattern, am_repl, sentence)
    return re.sub(pm_pattern, pm_repl, sentence)


def standardize_word_forms(sentence):
    """
    Replace words with a standardized version.
    >>> standardize_word_forms("Does dr smith teach any courses worth four credits?")
    'Does Dr. smith teach any courses worth 4 credits ?'
    """
    # TODO: make a JSON with the word-forms dict
    corrections = {"one": "1",
                   "two": "2",
                   "three": "3",
                   "four": "4",
                   "five": "5",
                   "six": "6",
                   "seven": "7",
                   "eight": "8",
                   "nine": "9",
                   "ten": "10",
                   "eleven": "11",
                   "twelve": "12",
                   "dr": "Dr.",
                   "Dr": "Dr.",
                   "dr.": "Dr.",
                   "Prof": "Professor",
                   "Professor": "Professor",
                   "prof.": "Professor",
                   "eecs": "EECS"
                   }
    tokens = nltk.word_tokenize(sentence)
    correct_tokens = []
    for word in tokens:
        if word in corrections:
            correct_tokens.append(corrections[word])
        else:
            correct_tokens.append(word)
    return " ".join(correct_tokens)