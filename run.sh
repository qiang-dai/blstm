#python3 step01_format_sentence.py > t
#python3 step11_save_index.py 5000000
#python3 step04_format_multi_punc.py raw_data/myfile.txt 1000000000 raw_data/res.txt
python3 step04_format_multi_punc.py raw_data/3.txt 1000000000 raw_data/res.txt
python3 step05_slip_window.py raw_data/res.txt 1000000000 raw_data/split_window.txt
python3 step11_save_index.py raw_data/split_window.txt 500000000
python3 step21_deep_learning.py 1000
#python3 step31_check_result.py 100 96
