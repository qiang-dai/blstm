## blstm 学习代码

## 执行步骤:
# 1,拉取代码
https://github.com/qiang-dai/blstm

# 2,进入代码目录，创建文件夹
- cd blstm
- mkdir raw_data/dir_kika/  raw_data/dir_subtitle/  raw_data/dir_twitter/  raw_data/dir_wiki/

# 3, 将4个训练文件，分布放入 4 个目录内

# 4, 修改./run.sh 将txt修改为预测文件
- 比如 python3 step41_file_predict.py valid_kika.txt > c1
- 改为 python3 step41_file_predict.py 1.txt > c1

# 5，执行 ./run.sh 开始训练

# 6，查看结果
- 准确率，召回率可以看文件 c1,c2,c3,c4
- 最终的预测结果内容，可以看 predict_result_valid_kika_cat0_.txt.txt (predict_result_1.txt.txt)
