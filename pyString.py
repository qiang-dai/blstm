#!/usr/local/python33/bin/python3
#-*- coding: gb18030 -*-

import os, sys, re

import full_chinese_character3
import pyUsage


def get_md5(s):
    try:
        import hashlib
        m2 = hashlib.md5()
    except ImportError:
        import md5
        m2 = md5.new()
    m2.update(s)
    return m2.hexdigest()

def get_diff_set(big_list, small_list):
    tmp_dict = dict([(e, True) for e in small_list])
    diff_list=[e for e in big_list if e not in tmp_dict]
    return diff_list

def getIndexLine(line): 
    t_list = line.split('\t')
    t_list = ['[%s]'%i + e for i,e in enumerate(t_list)]
    res = '\t'.join(t_list)
    return res

def isKanji(ch):
    try:
        t = ch.encode('gbk')
        if len(t) == 2:
            high = t[0]
            low = t[1]
            #print('h,l= 0x%X 0x%X'%(high, low))
            if 0xB0 <= high and high <= 0xF7:
                if 0xA1 <= low and low <= 0xFE:
                    return True
            if 0x81 <= high and high <= 0xA0:
                if 0x40 <= low and low <= 0xFE:
                    return True
            if 0xAA <= high and high <= 0xFE:
                if 0x40 <= low and low <= 0xA0:
                    return True
    except:
        return False
    return False
        
def isCharactorSeperator(ch):
    if ch in full_chinese_character3.half_characters_dict:
        return True
    if ch in full_chinese_character3.full_characters_dict:
        return True
    if ch in full_chinese_character3.other_characters_dict:
        return True

def isCharactorEnglish(ch):
    if ch in full_chinese_character3.full_eng_list:
        return True
    if ch in full_chinese_character3.half_eng_list:
        return True
    return False

def isCharactorNumber(ch):
    if ch in full_chinese_character3.full_digit_list:
        return True
    if ch in full_chinese_character3.half_digit_list:
        return True
    return False

def isAllKanji(text):
    for e in text:
        if not isKanji(e):
            return False
    return True
    
def isAllSeperator(text):
    for e in text:
        if not isCharactorSeperator(e):
            return False
    return True
    
def isAllEnglish(text):
    for e in text:
        if not isCharactorEnglish(e):
            return False
    return True
    
def isAllNumber(text):
    for e in text:
#         print(pyUsage.get_cur_info(), 'e= ', e)
#         print(pyUsage.get_cur_info(), text, '===')
        if not isCharactorNumber(e):
            return False
    return True
    
def isAllNumberOrEnglish(text):
    #for e in text:
    #for i, e in enumerate(text):
#     for i in range(len(text)):
#         print(pyUsage.get_cur_info(), 'i= ', i, 'text[i]= ', text[i])
        
    for i, e in enumerate(text):
#         print(pyUsage.get_cur_info(),'e= ', e, ' len(e)= ',  len(e), 'i= ', i, 'len(text)= ', len(text))
        if not isCharactorNumber(e) and not isCharactorEnglish(e):
#             print(pyUsage.get_cur_info(), 'e= ', e)
#             print(pyUsage.get_cur_info(), text, '===', e, ' is number:', isCharactorNumber(e), '; is eng: ', isCharactorEnglish(e))
            return False
    return True
        
def hasEnglishCharactor(word):
    for i in range(len(word)):
        ch = word[i]
        if isCharactorEnglish(ch):
            return True
    return False
    

def removeHeadTailSpace(word, flag = ' '):
    while len(word) > 0 and word[0] == flag:
        word = word[1:]
    while len(word) > 0 and word[-1] == flag:
        word = word[:-1]
    return word

def changeDoubleSpace2SingleSpace(text):
    while text.find('  ') != -1:
        text = text.replace('  ',' ')
        #print 'text= ',text
    return text

def reExtractData(regex, content, index):
    r = ''
    if index == None:
        index = 1
        
    p = re.compile(regex)
    m = p.search(content)
    #print('m.group()= ', m.group())
    if m != None: 
        #print('m.group()= ', m.group())
        r = m.group(index)
    return r

def my_spider_key(c):
    return c.encode('utf8')

def insert_or_add_dict(some_dict, key, value):
    if key in some_dict:
        some_dict[key] += value
    else:
        some_dict[key] = value

def mysort_k_cnt(e):
    k = float(e[0])
    return k

def mysort_k_cnt_reverse(e):
    k = float(e[0])
    return -1*k

def mysort_v(e):
    v = float(e[1])
    return v

def mysort_v_reverse(e):
    v = float(e[1])
    return -1*v

def sort_dict_by_key_cnt(some_dict, is_reverse = True):
    res_list = []
    for key in some_dict:
        val = some_dict[key]
        res_list.append((key, str(val)))
    if is_reverse:
        res_list.sort(key=mysort_k_cnt_reverse)
    else:
        res_list.sort(key=mysort_k_cnt)
    return res_list     

def sort_dict_by_cnt(some_dict, is_reverse = True):
    res_list = []
    for key in some_dict:
        val = some_dict[key]
        res_list.append((key, str(val)))
    if is_reverse:
        res_list.sort(key=mysort_v_reverse)
    else:
        res_list.sort(key=mysort_v)
    return res_list

def show_list(some_list):
    for e in some_list:
        #print '\t'.join(e)
        print (e[0]+'\t'+e[1])
        
###0x26020400 --- 637666304
###0x27000000 --- 638582784

###0x16020600 --- 369231360
###0x17020600 --- 386008576

def is_souyisou(cli_ver):
    try:
        if isinstance(cli_ver, str):
            cli_ver = int(cli_ver)
        if 369230882 <= cli_ver and cli_ver < 469230882:
            return True, 1
        if 637666358 <= cli_ver and cli_ver < 737666358:
            return True, 2
    except:
        pass
    return False, 0  

def insert_or_append_dict(some_dict, key, value):
    if key in some_dict:
        some_dict[key].append(value)
    else:
        some_dict[key] = [value, ]

def getLengthHan(e):
    length = 0
    for i in xrange(len(e)):
        if charactorIsUnicodeHan(e[i]):
            length += 2
        else:
            length += 1
    return length
            
def formatHan(some_list):
    max = 0
    for e in some_list:
        if max < getLengthHan(e):
            max = getLengthHan(e)
    
    ###2个空格
    max += 2
    ###
    t = []
    for e in some_list:
        #print get_cur_info(), e
        e += ' '*(max - getLengthHan(e))
        t.append(e)
    return t
    

def isEncoding(text, en):
    try:
        text.encode(en)###'gbk'
        return True
    except:
        pass
    return False

def fragmentList(text_list, frag_cnt):
    double_res_list = []
    #size = (len(text_list) + frag_cnt - 1) //frag_cnt
    size = len(text_list) //frag_cnt + 1
    print('size= ', size)

    for i in range(frag_cnt):
        beg = size * i
        end = size *(i+1)
        double_res_list.append(text_list[beg:end])
    for d in double_res_list:
        print ('len(d)= ', len(d))

    return double_res_list

def combineDoubleList2List(d_list):
    t_list = []
    [t_list.extend(e) for e in d_list]
    return t_list

if __name__ == '__main__':
    pass
    s = '14.98.17.26 - - [28/Jul/2016:16:44:04 +0800] "GET /v1/gif/search?query=%F0%9F%98%89&sign=83EDB9A4A07076EC5F54492426BC3983 HTTP/1.1" 454 200 200 "-" 4385 "Keyboard/2.5.5 (5A85B09A856D4C8A994905E027-16CE5/4a5ae1b93f21afaf199a6b3140e05097) Country/IN Language/en_IN System/iOS Version/9.3.3 Screen/iPhone6,2" "172.31.23.214" "172.31.17.229:9800" 0.006 0.006'
    str_duid = '[\-0-9a-zA-Z]+'
    reg = '\((%s)/(%s)\)'%(str_duid, str_duid)
    v = reExtractData(reg, s, 1)
    print ('v= ', v)
    #print split_size(1, 3)
    #print split_size(10, 3)
    #print split_size(100, 3)
    #print split_size(99, 3)
    #print split_size(98, 3)
    #print split_size(97, 3)
    #print split_size(1, 7)
    #print split_size(10, 7)
    #print split_size(100, 7)
    #print split_size(99, 7)
    #print split_size(98, 7)
    #print split_size(97, 7)
#     a='你好，我是中国人'
#     b='联通'
#     print ('len(a)= %d, len(b)= %d'%(len(a), len(b)))
#     print (a, b)
    
#    d = fragmentList(range(100), 3)
#    for e in d:
#        print ('len(e)= ', len(e))
