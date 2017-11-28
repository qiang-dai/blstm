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
    'LEFT'
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

def save_word_cnt(cnt):
    pyIO.save_to_file('%d'%cnt, 'raw_data/word_cnt.txt')

def get_word_cnt():
    with open('raw_data/word_cnt.txt', 'rb') as inp:
        cnt = inp.read().decode('utf8').strip()
        return int(cnt)

def save_punc_list(punc_list):
    pyIO.save_to_file('\t'.join(punc_list), 'raw_data/punc.txt')

def get_punc_list():
    with open('raw_data/punc.txt', 'rb') as inp:
        puncs = inp.read().decode('utf8').split('\t')
        return puncs


from emoji import UNICODE_EMOJI

def is_emoji(s):
    return s in UNICODE_EMOJI