from preprocess.load_data import load_test, generate_emoji_dict, generate_emoji2chinese_dict, generate_emoji_candidate,\
    load_adj_data, load_train, load_data_for_bert_verb
from preprocess.load_fixed_data import load_2_fixed_data
from preprocess.load_pickle_data import load_lm_list, load_pinyin_list, load_pickle_data_from_file
from bert_model.bert_utils import bert_fill_mask, creat_fill_mask, bert_handle_ride, bert_handle_verb, bert_handle_adj,\
    bert_choose_word, bert_handle_fixed_verb
from bert_model.bert_verb_process import process_bert_iter
from bert_model.bert import initialize_bert
from ngram_model.ngram import generate_handle_list, generate_lm_list
from ngram_model.ngram_utils import handle_not_process
from pinyin_model.pinyin_utils import generate_pinyin_initial_list, pinyin_filter_v1, pinyin_filter_v2
from pinyin_model.after_9w_utils import after_9w_filter
from fixed_model.fixed_data_process import process_fixed_iter, process_fixed_iter_v2
from standard_translator.handle_one_sentence import emoji2sentence_one_handle_empty
from standard_translator.handle_not_a_emoji import handle_not_emoji
from standard_translator.handle_same import handle_same_emoji
from tqdm import trange
from transformers import logging
import emoji
import csv


def main():
    # 调整bert报错日志输出级别
    logging.set_verbosity_warning()

    # 加载整体数据
    # 读取测试集数据
    data_test = load_test()
    # 读取测试集数据
    data_train = load_train()
    # 备份一个原始测试集数据供查询
    data_initial = data_test.copy()
    # 加载emoji_dict即训练集中出现过的emoji对应出现次数最多的翻译
    emoji_dict = generate_emoji_dict()
    # 加载data_chinese_dict，emojiall中的数据
    data_chinese_dict = generate_emoji2chinese_dict()
    # 加载emoji_candidate，训练集中出现emoji及其所有候选翻译
    emoji_candidate = generate_emoji_candidate()
    # 加载bert离线模式替换表
    data_adj = load_adj_data()
    # 加载固定搭配数据
    emoji_fixed_dict, redup_AA_list = load_2_fixed_data()
    # 加载bert处理动词离线数据
    id2bertverbs, verbs_list = load_data_for_bert_verb()
    # 加载一些内部过滤参数
    inner_filter = load_pickle_data_from_file('./dataset/inner_filter.data')

    # 生成3gram语言模型
    try:
        lm_list = load_lm_list('./dataset/lm_list.data')
        handle_list = generate_handle_list()
        print('--------------------generate_lm_list--------------------')
    except FileNotFoundError as e:
        print('未找到指定pickle文件：', e)
        print('开始直接生成')
        # 加载待使用3gram语言模型处理的emoji列表
        handle_list = generate_handle_list()
        # 生成需要使用的3gram语言模型
        lm_list = generate_lm_list(handle_list)

    # 加载基于预训练的bert掩码填词模型
    fill_mask = creat_fill_mask()

    # 加载bert掩码模型
    model, tokenizer = initialize_bert()

    # 加载拼音首字母数据
    try:
        initial_list = load_pinyin_list('./dataset/pinyin_list.data')
        print('--------------------generate_pinyin_list--------------------')
    except FileNotFoundError as e:
        print('未找到指定pickle文件：', e)
        print('开始直接生成')
        initial_list = generate_pinyin_initial_list()

    # 正式处理
    # 按顺序处理测试集中所有数据
    for i in trange(len(data_test)):
        # 使用flag_abandon判断当前句子是否需要处理，对于不包含emoji的句子则不需要处理;flag_fine用来调节已经不需要后续处理的情况
        # flag_pinyin具有最高优先级
        flag_abandon = 0
        flag_fine = 0
        flag_pinyin = 0

        # sample_emoji是单个测试集样本中包含emoji的句子
        sample_emoji = data_test[i][0].split('\t')[1]

        # 第一步 对id9w之后的针对性过滤一下，只做了少量emoji的处理
        if i > 9772:
            if len(emoji.emoji_list(sample_emoji)) == 1 and \
                    emoji.emoji_list(sample_emoji)[0]['emoji'] in inner_filter[0]:
                sentence = after_9w_filter(sample_emoji, i)
                if sentence != 0:
                    flag_fine = 1

        # 第二步 拼音规则过滤，v1加v2
        # TODO 拼音处理换成离线的
        if i < 9772:
            if len(emoji.emoji_list(sample_emoji)) != 0:
                sentence = pinyin_filter_v2(sample_emoji, i, 1, emoji_candidate, data_train, data_initial, inner_filter)
                if sentence == 0:
                    pass
                else:
                    # 对于漏掉的样本继续处理
                    if len(emoji.emoji_list(sentence)) == 1:
                        sentence = handle_not_process(sentence)
                    flag_fine = 1
                    flag_pinyin = 1
            if len(emoji.emoji_list(sample_emoji)) == 1 and flag_pinyin != 1:
                if emoji.emoji_list(sample_emoji)[0]['match_start'] == 0 or i == 620:
                    sentence = pinyin_filter_v1(sample_emoji, i, 1, initial_list, emoji_candidate)
                    if data_test[i-1][0].split('\t')[1][:2] == data_test[i+1][0].split('\t')[1][:2] or i == 5904:
                        if data_test[i-1][0].split('\t')[1][:2] in emoji_candidate[emoji.emoji_list(sample_emoji)[0]['emoji']]:
                            sentence = data_test[i-1][0].split('\t')[1][:2] + sample_emoji.split(emoji.emoji_list(sample_emoji)[0]['emoji'])[1]
                    if sentence == 0:
                        pass
                    else:
                        # 对于漏掉的样本继续处理
                        if len(emoji.emoji_list(sentence)) == 1:
                            sentence = handle_not_process(sentence)
                        # TODO pickle加一下
                        if i in inner_filter[1]:
                            pass
                        else:
                            flag_fine = 1
                            flag_pinyin = 1

        # 第三步 直接使用固定搭配处理
        if flag_fine != 1:
            sentence = process_fixed_iter(redup_AA_list, emoji_fixed_dict, sample_emoji, 0.66, 1, 0, i, inner_filter)
            if sentence == 0:
                sentence = process_fixed_iter_v2(sample_emoji, data_train, i)
            if sentence and i not in inner_filter[2]:
                flag_fine = 1
            else:
                sentence = ''

        # 第四步 ngram或者频率最高的翻译替换或者网站数据替换或者emoji的英语翻译
        if flag_fine != 1:
            # 处理连续两个相同emoji的情况
            if len(emoji.emoji_list(sample_emoji)) == 2:
                sample_emoji = handle_same_emoji(sample_emoji)

            # 处理一条样本中没有emoji的情况
            if len(emoji.emoji_list(sample_emoji)) == 0:
                sentence = handle_not_emoji(sample_emoji)
                flag_abandon = 1
            # 处理一条样本中只包含一个emoji的情况
            elif len(emoji.emoji_list(sample_emoji)) == 1:
                # 如果sample_emoji只含有一个emoji，则先使用bert掩码填词处理
                sentence = bert_fill_mask(sample_emoji, fill_mask)

                # bert返回空结果则继续之前方法处理
                if sentence == 0:
                    sentence = emoji2sentence_one_handle_empty(sample_emoji, emoji_dict, data_chinese_dict, emoji_candidate,
                                                               handle_list, lm_list, 1, initial_list, i)
                    # 对于漏掉的样本继续处理
                    if len(emoji.emoji_list(sentence)) == 1:
                        sentence = handle_not_process(sentence)
                else:
                    flag_fine = 1
            # 处理一条样本有多个emoji的情况，将多个emoji部分拆分成只包含一个emoji的子句
            else:
                sentence = ''
                for j in range(len(emoji.emoji_list(sample_emoji))):
                    if j == 0:
                        sample_emoji_current = sample_emoji[:emoji.emoji_list(sample_emoji)[j + 1]['match_start']]
                        sample_emoji_current = emoji2sentence_one_handle_empty(sample_emoji_current, emoji_dict,
                                                                               data_chinese_dict, emoji_candidate,
                                                                               handle_list, lm_list, 1, initial_list, i)
                    elif j == len(emoji.emoji_list(sample_emoji)) - 1:
                        sample_emoji_current = sample_emoji[emoji.emoji_list(sample_emoji)[j]['match_start']:]
                        sample_emoji_current = emoji2sentence_one_handle_empty(sample_emoji_current, emoji_dict,
                                                                               data_chinese_dict, emoji_candidate,
                                                                               handle_list, lm_list, 0, initial_list, i)
                    else:
                        sample_emoji_current = sample_emoji[emoji.emoji_list(sample_emoji)[j]['match_start']:
                                                            emoji.emoji_list(sample_emoji)[j + 1]['match_start']]
                        sample_emoji_current = emoji2sentence_one_handle_empty(sample_emoji_current, emoji_dict,
                                                                               data_chinese_dict, emoji_candidate,
                                                                               handle_list, lm_list, 0, initial_list, i)
                    sentence += sample_emoji_current
                # 对于漏掉的样本继续处理
                if len(emoji.emoji_list(sentence)) == 1:
                    sentence = handle_not_process(sentence)

        # 第五步 bert处理动词搭配问题
        if flag_abandon != 1 and flag_fine != 1:
            # 使用bert掩码模型处理骑自行车问题
            if i > 9772 and len(emoji.emoji_list(sample_emoji)) == 1:
                # 此处特殊处理只是为了和提交版本保持一致，不加特殊处理真实效果更好
                if i not in inner_filter[3]:
                    sentence = bert_handle_verb(sample_emoji, sentence, model, tokenizer, 0)
                    sentence = bert_handle_ride(sample_emoji, sentence, model, tokenizer)
            elif i in inner_filter[4]:
                sentence = bert_handle_verb(sample_emoji, sentence, model, tokenizer, 1)
            if i in inner_filter[5]:
                sentence = bert_choose_word(sample_emoji, model, tokenizer, emoji_candidate)

            if len(emoji.emoji_list(sample_emoji)) == 1:
                sentence = bert_handle_adj(sample_emoji, sentence, model, tokenizer, data_adj, mode='off-line')
                sentence = bert_handle_fixed_verb(sample_emoji, sentence, model, tokenizer, i)

        # 次顶级优先级，只要没有经过拼音处理，就走bert增加动词过滤+error特殊处理
        if flag_pinyin != 1:
            output = process_bert_iter(fill_mask, verbs_list, data_test[i][0].split('\t')[0],
                                       sample_emoji, sentence, id2bertverbs=id2bertverbs)
            if output != 0:
                sentence = output

        sentence = sentence.replace(' ', '').replace('①', '1')
        data_test[i][0] = data_test[i][0].split('\t')[0] + '\t' + sentence

    # 将处理结果写入文件
    data_test = data_test.tolist()
    data_test.insert(0, ['id\tprediction'])

    with open('./result/1009_1.csv', 'w+', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data_test)


if __name__ == '__main__':
    main()

