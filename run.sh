#python3 step01_format_sentence.py > t
#python3 step11_save_index.py 1000000
python3 step04_format_multi_punc.py raw_data/en_punctuation_recommend_train_100W 10000000 raw_data/res.txt
python3 step11_save_index.py raw_data/res.txt 1000000
python3 step21_deep_learning.py 1000
python3 step31_check_result.py 100 96
