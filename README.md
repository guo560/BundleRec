1. main.py中含有4个run_xxx_main函数，分别表示4个数据集对应的模型训练函数
2. run_xxx_main函数中的各个超参在run_bundle_main中有注释
3. 注意BST中TimeEmbedding类中对于timestamps的变换，饿了么数据集中的timestamps是到当前推荐的时间间隔，其它数据集均为时间戳
4. 原始数据集在./data/raw_data路径下
5. 处理后的数据集分别在./data/bundle_time，./data/ele_time，./data/movielens1m，./data/taobao
6. 处理后的数据集中包含lbes.pkl、mms.pkl、train_data.csv、val_data.csv、test_data.csv这5个文件