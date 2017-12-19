import pyIO

punctuation_all_list = [
    'SP',
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
    '=',
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

punctuation_list = [
    "SP",
#    "...",
#    "{",
#    "}",
#    "[",
#    "]",
#     ")",
#     "(",
#    ">",
#    "<",
#    "#",
#    "@",
#    "+",
#    "-",
#    "*",
#    "%",
#    '=',
#    '$',
#    '&',
#    '~',
#     ";",
#    "^",
#    "_",
#    "\"",
#    '/',
#    '\\',
    "!",
    "?",
    ":",
    "'",
    ".",
    ',',
    'OTHER'
]


def getCharType(c):
    if is_alphabet(c):
        return 1
    if is_number(c):
        return 2
    if is_punc(c):
        return 3
    if is_emoji(c):
        return 4
    return 5

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

punc_set = set(punctuation_all_list)
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

###其他所有标点符号
def get_punc_other():
    return 'OTHER'
def get_punc_filled():
    return punctuation_list[0]
def get_punc_set():
    return set(punctuation_all_list)

def is_valid_punc(punc):
    #if len(punc) != 1:
    #    return False
    if punc in punctuation_list:
        return True
    return False

def get_punc_list():
    with open('raw_data/punc.txt', 'rb') as inp:
        puncs = inp.read().decode('utf8').split('\t')
        return puncs


from emoji import UNICODE_EMOJI

def is_emoji(s):
    return s in UNICODE_EMOJI

def get_header_word():
    return 'Header'

def get_tail_word():
    return 'Tail'

def get_filled_word():
    return 'NONE'

def get_batch_size():
    return 128
    #return 4

def get_timestep_size():
    #return 64
    #return 32
    return 8;