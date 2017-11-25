import pyIO

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

def save_punc_list(punc_list):
    pyIO.save_to_file('\t'.join(punc_list), 'raw_data/punc.txt')

def get_punc_list():
    with open('raw_data/punc.txt', 'rb') as inp:
        puncs = inp.read().decode('utf8').split('\t')
        return puncs