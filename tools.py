import sys,os,time

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
