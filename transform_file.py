#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys,os
import json,pyString
import pyIO
import punctuation

###解析 monitor.log
if __name__ == '__main__':
    filename = sys.argv[1]
    t_list = pyIO.get_content(filename)

    res_list = []
    for t in t_list:
        r = ''
        for c in t:
            if c in punctuation.get_punc_list():
                if c == '?':
                    c = ':'
                else:
                    c = '.'
            r += c
        res_list.append(r)
    pyIO.save_to_file('\n'.join(res_list), filename.replace('.txt', '_res.txt'))