import argparse
import emoji
from pinyin_model.after_9w_utils import after_9w_filter
from fixed_model.fixed_data_process import process_fixed_iter
from preprocess.load_fixed_data import load_2_fixed_data
from preprocess.load_pickle_data import load_pickle_data_from_file
from preprocess.load_data import load_data_for_bert_verb
from standard_translator.handle_one_sentence import general_process
from bert_model.bert_utils import bert_fill_mask, creat_fill_mask, bert_handle_ride, bert_handle_verb, bert_handle_adj
from bert_model.bert_verb_process_epoch import process_bert_iter
from bert_model.bert import initialize_bert


def translate_a_sentence(opt):
    """
    命令行api，翻译单条句子
    :param opt: 参见本文件下方参数提示或使用-h
    :return: 翻译好的句子
    """
    sentence = opt.sentence
    init_sentence = opt.sentence
    if len(emoji.emoji_list(sentence)) == 0:
        print('no emoji found in sentence')
        return sentence

    if opt.context == 'normal':
        result = after_9w_filter(sentence)
        if result:
            return result

    if opt.fixed:
        emoji_fixed_dict, redup_AA_list = load_2_fixed_data()
        inner_filter = load_pickle_data_from_file('./dataset/inner_filter.data')
        result = process_fixed_iter(redup_AA_list, emoji_fixed_dict, sentence, 0.66, 1, 0, 0, inner_filter)
        if result:
            return result

    general_result = general_process(sentence)

    if opt.bert:
        fill_mask = creat_fill_mask()
        model, tokenizer = initialize_bert()
        id2bertverbs, verbs_list = load_data_for_bert_verb()
        # bert固定搭配单MASK填空
        result = bert_fill_mask(sentence, fill_mask)
        if result:
            return result
        # bert添加一般动词检测
        result = process_bert_iter(fill_mask, verbs_list, init_sentence, sentence)
        if result:
            return result
        # bert添加一般动词检测v2
        result = bert_handle_verb(init_sentence, sentence, model, tokenizer)
        if result:
            return result
        # bert交通工具类动词检测
        result = bert_handle_ride(init_sentence, sentence, model, tokenizer)
        if result:
            return result
        # bert形容词检测
        result = bert_handle_adj(init_sentence, sentence, model, tokenizer, [], mode='on-line')
        if result:
            return result

    return general_result


def translate_a_sentence_inner(sample_emoji, context, fixed, bert):
    """
    翻译单条带有emoji的句子
    :param sample_emoji: 带有emoji的句子
    :param context: 当前语境，适用于抖音黑灰产语境和一般语境[douyin|normal]
    :param fixed: 是否使用固定搭配模型，默认为使用
    :param bert: 是否使用bert模型，默认为使用（处理较慢）
    :return: 翻译好的句子
    """
    sentence = sample_emoji
    init_sentence = sample_emoji
    if len(emoji.emoji_list(sentence)) == 0:
        print('no emoji found in sentence')
        return sentence

    if context == 'normal':
        result = after_9w_filter(sentence)
        if result:
            return result

    if fixed:
        emoji_fixed_dict, redup_AA_list = load_2_fixed_data(log=False)
        inner_filter = load_pickle_data_from_file('./dataset/inner_filter.data')
        result = process_fixed_iter(redup_AA_list, emoji_fixed_dict, sentence, 0.66, 1, 0, 0, inner_filter)
        if result:
            return result

    general_result = general_process(sentence, log=False)

    if bert:
        fill_mask = creat_fill_mask(log=False)
        model, tokenizer = initialize_bert()
        id2bertverbs, verbs_list = load_data_for_bert_verb(log=False)
        # bert固定搭配单MASK填空
        result = bert_fill_mask(sentence, fill_mask)
        if result:
            return result
        # bert添加一般动词检测
        result = process_bert_iter(fill_mask, verbs_list, init_sentence, general_result)
        if result:
            return result
        # bert添加一般动词检测v2
        result = bert_handle_verb(init_sentence, general_result, model, tokenizer)
        if result:
            return result
        # bert交通工具类动词检测
        result = bert_handle_ride(init_sentence, general_result, model, tokenizer)
        if result:
            return result
        # bert形容词检测
        result = bert_handle_adj(init_sentence, general_result, model, tokenizer, [], mode='on-line')
        if result:
            return result

    return general_result


if __name__ == '__main__':
    """
    *Tips*
    本程序能够处理一般的含emoji文本翻译问题，其中使用到的主要技术包括ngram模型、固定搭配模型和bert语言模型等。
    本程序是对竞赛中使用的模型做了泛用性升级之后的程序，若直接将本程序用在竞赛测试集上将会得到84分左右的结果（差了拼音检测模块和语境针对性处理，
    详情参见Readme），若您想对竞赛测试集进行验证，请运行main_test_dataset.py
    """
    # python程序模式
    print(translate_a_sentence_inner('人生要往前看、生命一片好光景、💪一切都会越来越好的', 'douyin', 1, 1))
    print(translate_a_sentence_inner('退伍证可以免费坐🚌和🚇我一直在用', 'douyin', 1, 1))
    print(translate_a_sentence_inner('美国侵略📏拉克炸中国大使馆时你干啥去了', 'douyin', 1, 1))

    # 命令行模式
    # （命令行可能会出现emoji编码问题。若要使用请打开下方注释，使用方法参见python main_handle_a_sentence.py -h）
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--sentence', required=True, type=str,
    #                     help='emoji sentences waiting for translating.')
    # parser.add_argument('--context', type=str, default='douyin',
    #                     help='select the emoji sentence\'s context.[douyin(defualt)|normal]')
    # parser.add_argument('--fixed', type=bool, default=True,
    #                     help='whether to use the fixed matching module.[True(default)|False]')
    # parser.add_argument('--bert', type=bool, default=True,
    #                     help='whether to use bert module to enhance the translation model.[True(default)|False]')
    #
    # opt = parser.parse_args()
    #
    # translate_a_sentence(opt)



