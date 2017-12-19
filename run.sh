mkdir raw_data/dir_step00/ raw_data/dir_step03/ raw_data/dir_step04/ raw_data/dir_step07/ 
mkdir tmp/ data/ ckpt/
rm -f ckpt/*
rm -f raw_data/dir_step00/*
rm -f raw_data/dir_step03/*
rm -f raw_data/dir_step04/*
rm -f raw_data/dir_step07/*
rm -f tmp/step*/*
rm -f tmp/*txt
rm -f tmp/*pkl
rm -f data/*
python3 step01_format_kika.py       raw_data/dir_kika       1000
python3 step01_format_subtitle.py   raw_data/dir_subtitle   1000
python3 step01_format_twitter.py    raw_data/dir_twitter    1000
python3 step01_format_wiki.py       raw_data/dir_wiki       1000

python3 step03_split_file.py raw_data/dir_step00            5000              raw_data/dir_step03
python3 step04_format_multi_punc.py raw_data/dir_step03     10000000000       raw_data/dir_step04     1
python3 step07_slip_window.py raw_data/dir_step04          1000000000        raw_data/dir_step07

python3 step51_fastText_classify.py  train
python3 step23_saver_learning_multi.py                       raw_data/dir_step07             2
python3 step41_predict.py       p.txt

