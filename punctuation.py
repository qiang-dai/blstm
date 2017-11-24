punctuation_list = [
    'UNKNOWN',
    "...",
    "{",
    "}",
    "[",
    "]",
    ")",
    "(",
    ">",
    "<",
    "#",
    "@",
    "+",
    "-",
    "*",
    "%",
    '$',
    '&',
    '~',
    "!",
    "?",
    "_",
    ":",
    ";",
    "^",
    "'",
    "\"",
    ".",
    ',',
    '/',
    '\\',
]

def is_chinese(uchar):
    if u'\u4e00' <= uchar<=u'\u9fff':
        return True
    else:
        return False

def is_number(uchar):
    if u'\u0030' <= uchar and uchar<=u'\u0039':
        return True
    else:
        return False

def is_alphabet(uchar):
    if (u'\u0041' <= uchar <= u'\u005a') or (u'\u0061' <= uchar <= u'\u007a'):
        return True
    else:
        return False

punc_set = set(punctuation_list)
def is_punc(uchar):
    if uchar in punc_set:
        return True
    else:
        return False