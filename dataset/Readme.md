## ./dataset/
## *tips:请自行下载[bert_base_chinese预训练模型](https://huggingface.co/bert-base-chinese/tree/main)，并将文件夹放在./dataset/中*

该目录下包含了本项目所需要的所有数据  
1.  ./bert_base_chinese/存放bert预训练模型（**需自行下载**）  
文件夹中应至少包括pytorch_model.bin(392MB)和该模型的配置文件以及tokenizer文件（tf和flax版本不需要）
2.  ./verb_and_fixed_data/存放bert动词模型和固定搭配模型需要用到的预处理好的数据
3.  _adj_bert_change.csv存放bert处理形容词之后的替换表，可以增加测试集处理速度（也可不使用，设置mode为online）
4.  chinese_verb.txt是网上获取的汉字常见动词表
5.  emoji2chinese.csv是在emoji中文网爬取的单emoji中文翻译数据
6.  emoji7w.csv是赛事提供的训练集
7.  emoji7w-test_data.csv是赛事提供的测试集
8.  emoji_dict_4_columns.csv是包含从训练集提取的单emoji候选翻译即其出现次数和单emoji标准英语翻译的文件
9.  inner_filter.data使用pickle固定的测试集处理过滤参数
10. lm_list.data存放预处理好的ngram模型（也可不使用，文件不存在则会自动生成）
11. pinyin_list.data存放使用pickle固定的测试集文本首字拼音，用于拼音模块处理、
12. **submit_result.csv比赛最终提交的测试集翻译版本**
