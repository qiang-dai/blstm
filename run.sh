limit_cnt=$1
echo 'limit_cnt:'${limit_cnt}

# sudo pip3 install tqdm
# sudo pip3 install fasttext
# sudo pip3 install cython
# sudo pip3 install fasttext
# sudo pip3 install emoji

mkdir raw_data/dir_kika/  raw_data/dir_subtitle/  raw_data/dir_twitter/  raw_data/dir_wiki/
cp raw_data/en_punctuation_recommend_train_100W raw_data/dir_kika/
mkdir raw_data/dir_step00/ raw_data/dir_step03/ raw_data/dir_step04/ raw_data/dir_step07/
mkdir tmp/ data/ ckpt/
mkdir tmp/step01/ tmp/step04/ tmp/step07/

rm -f ckpt/*
rm -f raw_data/dir_step00/*
rm -f raw_data/dir_step03/*
rm -f raw_data/dir_step04/*
rm -f raw_data/dir_step07/*
rm -f tmp/step*/*
rm -f tmp/*txt
rm -f tmp/*pkl
rm -f data/*
python3 step01_format_kika.py       raw_data/dir_kika       ${limit_cnt}
python3 step01_format_subtitle.py   raw_data/dir_subtitle   ${limit_cnt}
python3 step01_format_twitter.py    raw_data/dir_twitter    ${limit_cnt}
python3 step01_format_wiki.py       raw_data/dir_wiki       ${limit_cnt}

python3 step03_split_file.py raw_data/dir_step00            500000              raw_data/dir_step03
python3 step04_format_multi_punc.py raw_data/dir_step03     100000       raw_data/dir_step04     1
python3 step07_slip_window.py raw_data/dir_step04           1000000        raw_data/dir_step07

python3 step51_fastText_classify.py  train
wget http://nlp.stanford.edu/data/glove.twitter.27B.zip
unzip -of glove.twitter.27B.zip
cp glove.twitter.27B.100d.txt model_word.vec

python3 step23_saver_learning_multi.py                       raw_data/dir_step07             1

python3 step41_file_predict.py valid_kika.txt >> c1
python3 step41_file_predict.py valid_subtitle.txt >> c2
python3 step41_file_predict.py valid_twitter.txt >> c3
python3 step41_file_predict.py valid_wiki.txt >> c4
