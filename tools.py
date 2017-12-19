import sys,os,time
import pyIO
import datetime
import codecs
import punctuation

def args():
    filename = 'raw_data/total_english.txt'
    if len(sys.argv) > 1:
        filename = sys.argv[1]

    threshold_line_cnt = 10000
    if len(sys.argv) > 2:
        threshold_line_cnt = int(sys.argv[2])

    res_file = 'raw_data/total_english.txt'
    if len(sys.argv) > 3:
        res_file = sys.argv[3]
    return filename, threshold_line_cnt, res_file

def get_filename_list(file_dir):
    print('tools file_dir:', file_dir)
    filename_list = []
    if os.path.isfile(file_dir):
        print('tools is file, file_dir:', file_dir)
        filename_list.append(file_dir)
    else:
        print('tools is dir, file_dir:', file_dir)
        filename_list,_ = pyIO.traversalDir(file_dir)
    filename_list = [e for e in filename_list if e.find('DS_Store') == -1]
    return filename_list

def read_file_content_en(filename, encoding = 'utf8', ignore_empty = True, \
                         cur_cnt = 0, limit_cnt = 2000*10000):
    ###验证
    if cur_cnt >= limit_cnt:
        return []

    result_list = []
    fr = codecs.open(filename, 'r', encoding, 'ignore')
    index = 0;
    #print ('filename= ', filename)
    for text in fr:
        index += 1
        text = text.strip('\n')
        text = text.strip()
        if len(text) > 0 and text[0] != '#':
            result_list.append(text)
        elif not ignore_empty:
            result_list.append(text)
        ###
        if cur_cnt + len(result_list) > limit_cnt:
            break

    return result_list

def get_total_limit_list(file_dir, threshold_line_cnt):

    ###获得所有文件
    filename_list = get_filename_list(file_dir)

    ###所有结果
    total_list = []
    for filename in filename_list:
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'filename:', filename)

        #c_list = pyIO.get_content(filename)
        c_list = read_file_content_en(filename, 'utf8', True, len(total_list), threshold_line_cnt)
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'len(c_list):', len(c_list))

        ###按行进行添加
        for i, c in enumerate(c_list):
            if i%10000 == 0:
                print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'i:', i)
            total_list.append(c)

            if len(total_list) > threshold_line_cnt:
                break
        if len(total_list) > threshold_line_cnt:
            break
    return total_list

def add_space_inside_line(line):
    res = ''
    last_type = None
    for i, c in enumerate(line):
        cur_type = punctuation.getCharType(c)
        if last_type != cur_type:
            last_type = cur_type
            res += ' '
        res += c
    res = res.replace('  ', ' ')
    res = res.replace('  ', ' ')
    res = res.lower()
    return res