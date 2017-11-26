#!/usr/bin/env python
#-*- coding: gb18030 -*-

import codecs
import sys
import os
#import pyLog
import pyString
import shutil
import threading
import fcntl
import datetime
try:
    import commands
    import Queue
except:
    pass

#import os
#os.path.basename('/Volumes/1.mp4')   #输出('1.mp4')
#os.path.dirname('/Volumes/1.mp4')   #输出('/Volumes')
#os.path.splitdrive('Volumes/1.mp4')   #输出('','/Volumes/1.mp4')
#os.path.split('/Volumes/1.mp4')    #输出（‘/Volumes’，‘1.mp4’）
#os.path.split('/Volumes/text')     #输出（‘/Volumes’，‘text’）
#fname, fextension=os.path.splitext('/Volumes/Leopard/Users/Caroline/Desktop/1.mp4')
#print fname,fextension   #输出/Volumes/Leopard/Users/Caroline/Desktop/1 .mp4
#os.path.join('a','b','1.mp4')  #输出#‘a/b/1.mp4’

def is_machine_test2():
    cmd = "/sbin/ifconfig | grep 172.31.47.255"
    ret,out = commands.getstatusoutput(cmd)
    t = ''.join(out)
    #print 't= ', t
    if t.find('172.31.47.255') != -1:
        return True
    return False

def is_porc_run(proc_name):
    cmd = 'ps x | grep %s '%proc_name
    ret,out = commands.getstatusoutput(cmd)
    #print 'out=', ''.join(out)
    t_list = ''.join(out).split('\n')
    cnt = 0
    for i,t in enumerate(t_list):
        t = t.strip()
        #print 't= ', t
        if t.find('grep ') != -1:
            continue
        if t.find('sh -c ') != -1:
            continue
        if t.find('/bin/sh -c ') != -1:
            continue
        if t.find(proc_name) != -1:
            cnt += 1
    #print 'cnt= ', cnt
    if cnt > 1:
        return True
    return False
    
def create_dir_if_needed(filename):
    d,n = os.path.split(filename)
    if len(d) > 0:
        createDir(d)
###utf8 or binary
def save_to_file(text, filename):
    create_dir_if_needed(filename)
    f = codecs.open(filename, 'wb', 'utf8')
    #f = codecs.open(filename, 'w')
    f.write(text)
    f.close()

def lock_file(filename):
    fp = open(filename, 'w')
    fcntl.flock(fp, fcntl.LOCK_EX)  
    return fp

def unlock_file(fp):
    fcntl.flock(fp, fcntl.LOCK_UN) 

def append_to_file_nolock(text= '', filename='log.txt'):
    create_dir_if_needed(filename)
    f = open(filename, 'a')
    f.write(text)
    f.close()

g_mutex = threading.Lock()
def lock(mutex = None):
    global g_mutex
    cur_mutex = mutex
    if cur_mutex is None:
        cur_mutex = g_mutex
        cur_mutex.acquire()
    return cur_mutex

def unlock(cur_mutex):
    cur_mutex.release()

def append_to_file(text= '', filename='log.txt', mutex = None):
    create_dir_if_needed(filename)
    global g_mutex
    cur_mutex = mutex
    if cur_mutex is None:
        cur_mutex = g_mutex

    cur_mutex.acquire()

    f = codecs.open(filename, 'a', encoding = 'utf8')
    f.write(text)
    f.close()

    #g_mutex.release()
    cur_mutex.release()

def createDir(dst_dir):
    if not os.path.exists(dst_dir):
        try:
            os.makedirs(dst_dir)
        except BaseException as e:
            print ('ERROR e= ', e)

def clear_dir(dst_dir):
    if os.path.isdir(dst_dir):
        shutil.rmtree(dst_dir, ignore_errors=True)
    createDir(dst_dir)

def rm_file(filename):
    if os.path.exists(filename):
        os.remove(filename)

def add_time_suffix(src_filename):
    suf = datetime.datetime.now().strftime('%y%m%d_%H%M%S');
    pos2 = src_filename.find('/')
    pos = src_filename.rfind('.')
    if pos <= pos2:
        pos = len(src_filename)
    return src_filename[:pos]+'_'+suf + src_filename[pos:]

def rename_file(src_filename, dst_filename):
    if not os.path.exists(src_filename):
        return False
    if os.path.exists(dst_filename):
        return False
    shutil.move(src_filename, dst_filename)    

def copy_file(src_filename, dst_filename):
    if os.path.exists(src_filename):
        create_dir_if_needed(dst_filename)
        shutil.copy(src_filename, dst_filename)

def clear_file(filename):
    create_dir_if_needed(filename)
    fw = open(filename, 'w')
    fw.close()
        
def read_file_content_en(filename, encoding = 'utf8', ignore_empty = True):
    result_list = []
    fr = codecs.open(filename, 'r', encoding)
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
    return result_list

en_list = ['utf-8','cp936','gb18030','utf16', 'big5','euc-jp','euc-kr','latin1',]
def get_content(filename, encoding=None, ignore_empty = True):
    if not os.path.exists(filename):
        pyLog.logging.error('error! not exists file: %s'%filename)
        print('error! not exists file: %s'%filename)
        sys.exit(0)

    #####    
    if encoding is not None:
        c_list = read_file_content_en(filename, encoding, ignore_empty = ignore_empty)
        return c_list    
        
    #####
    c_list = []
    for i,en in enumerate(en_list):
        try:
            c_list = read_file_content_en(filename, en, ignore_empty = ignore_empty)
            break
            pass
        except BaseException as e:
            print ('ERROR e= ', e)
        if i == len(en_list) - 1:
            print ('error! failed read:', filename)

    return c_list

def traversalDir(path, is_recursive = True, is_add_link = True):
    file_list = []
    dir_list = []
    ###check permission
    try:    
        for i in os.listdir(path):
            sub_path = path + os.sep + i 
            sub_path = sub_path.replace(os.sep + os.sep, os.sep)
            if os.path.isfile(sub_path):
                file_list.append(sub_path)
            elif os.path.isdir(sub_path):# and is_recursive:
                ###ignore link file
                if os.path.islink(sub_path) and not is_add_link:
                    continue

                dir_list.append(sub_path)
                pass
                if is_recursive:
                    sub_file_list, sub_dir_list = traversalDir(sub_path)
                    file_list.extend(sub_file_list)
                    dir_list.extend(sub_dir_list)
                    pass
    except BaseException as e:
        print ('ERROR e= ', e)
    return file_list, dir_list

def is_modify_inside_time(pathname, diff = 60):
    #print 'pathname= ', pathname
    try:
        t = os.stat(pathname)[stat.ST_MTIME]
        cur = time.time()
        #print 't,cur= ', t, cur
        if cur - t < diff:
            return True
    except BaseException as e:
        print ('[pyIO.py 231]ERROR e= ', e)
    return False

def generateFileByCurrentScripyName(label):
    current_file = os.path.realpath(sys.argv[0])
    pos = current_file.rfind(os.sep)
    pos02 = current_file.rfind('.')
    path = current_file[:pos]
    filename = current_file[pos + 1:pos02]
    suf = current_file[pos + 1:]
    
    label = pyString.removeHeadTailSpace(label)
    if len(label) == 0:
        pyLog.logging.info('error! label is 0')
        sys.exit(0)
    
    return path, filename+label

def get_md5(c):
    try:
        import hashlib
        m2 = hashlib.md5()
    except ImportError:
        import md5
        m2 = md5.new()
    m2.update(s)
    return m2.hexdigest()

def get_file_md5(filename):
    ###md5
    if os.path.getsize(filename) < 1:#10*1024*1024:
        f = open(filename, 'rb')
        c = f.read()
        f.close()
        return get_md5(c)
    else:
        if os.path.exists('/sbin/md5'):
            cmd = '/sbin/md5 "video_filename"'###Mac
            cmd = cmd.replace('video_filename', filename)
            ret,out = commands.getstatusoutput(cmd)
            md5v = ''.join(out)
            md5v = md5v.strip()
            md5v = md5v.split(' ')[-1]
        else:
            cmd = 'md5sum "video_filename"'###Linux
            cmd = cmd.replace('video_filename', filename)
            ret,out = commands.getstatusoutput(cmd)
            md5v = ''.join(out)
            #print 'md5v= ', md5v
            md5v = md5v.strip()
            md5v = md5v.split(' ')[0]
        return md5v

def FormatEncoding(IN_FILE, IN_ENCODING, OUT_FILE, OUT_ENCODING):
    fp = open(IN_FILE, 'rb')
    index = 0
    text_list = []
    while True:
        eachline = fp.readline()
        if not eachline:
            break

        ###打印信息
        index += 1
        if (index + 1) % 10000 == 0:
            print ('index= %d'%index)
        try:
            text = eachline.decode(IN_ENCODING, 'replace')
            text.encode(OUT_ENCODING)
            text_list.append(text)
        except BaseException as e:
            print ('ERROR e= ', e)

    f = codecs.open(OUT_FILE, 'w', OUT_ENCODING)
    f.write(''.join(text_list))
    f.close()
    
import filecmp 
def isFileEqual(file01, file02):
    ###相等返回True
    return filecmp.cmp(file01, file02)

def currentDirFile():
    current_file = os.path.realpath(sys.argv[0])
    pos = current_file.rfind(os.sep)
    current_dir = current_file[:pos + 1]
    current_file = current_file[pos + 1:]
    return current_dir, current_file

def split_size(size, num = 3):
    #print 'size= ', size
    if size < num:
        return [size,]

    avg = size/num
    more = size%num
    res_list = []
    for i in range(num):
        if i < more:
            res_list.append(avg+1)
        else:
            res_list.append(avg)
    return res_list

def split_file_lines(filename, num, new_file = 'res'):
    c_list = get_content(filename, ignore_empty = False)
    #print filename, len(c_list)

    res_list = split_size(len(c_list), num)
    beg = 0
    for i,v in enumerate(res_list):
        end = beg+v

        CTRL = '\n'
        if i+1 == len(res_list):
            CTRL = ''

        save_to_file('\n'.join(c_list[beg:end]) + '\n', new_file + '.%d.txt'%i)
        beg = end

###for python2.7
def init_timeout_lock():
    timeout_queue = Queue.Queue()
    timeout_queue.put(1,block=False)
    return timeout_queue 

def timeout_lock(timeout_queue, timeout):
    #print 'timeout_queue.qsize()= ', timeout_queue.qsize()
    assert(timeout_queue.qsize() == 0 or timeout_queue.qsize() == 1)
    try:
        ret = timeout_queue.get(block=True, timeout=timeout)
        #print 'ret= ', ret
    except BaseException as e:
        print ('ERROR e= ', e)
        return False
    return True

def timeout_unlock(timeout_queue):
    assert(timeout_queue.qsize() == 0)
    timeout_queue.put(1,block=False)

def merge_file(filename, num, new_file= "res.txt"):
    fw = open(new_file, 'wb')
    for i in range(num):
        cur_filename = filename + '.' + '%s'%i
        f = open(cur_filename, 'rb')
        c = f.read()
        f.close()
        fw.write(c)
    fw.close()

def split_file(filename, num, new_file = "res.txt"):
    if not os.path.exists(filename):
        print ('file not exists:', filename)
        return
    size = os.path.getsize(filename)
    res_list = split_size(size, num)
    fp = open(filename, 'rb')
    beg = 0
    print ('res_list= ', res_list)
    for i,v in enumerate(res_list):
        fp.seek(beg, 0)
        c = fp.read(v)

        fw = open(new_file + '.%s'%i, 'wb')
        fw.write(c)
        fw.close()

        beg += v
if __name__ == '__main__':
    #split_file_lines(sys.argv[1], int(sys.argv[2]))
    #print is_porc_run('pyIO.py')
    #print is_machine_test2()
    #split_file(sys.argv[1], int(sys.argv[2]), sys.argv[3])
    pass
