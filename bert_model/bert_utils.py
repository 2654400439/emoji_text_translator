from preprocess.load_data import generate_emoji_candidate, load_test
from bert_model.bert import initialize_bert
import emoji
from transformers import pipeline, logging
import torch
from torch import nn
import numpy as np
import difflib
import jieba
from jieba import posseg


def bert_fill_mask(sample_emoji, fill_mask):
    """
    BERT掩码填词置信度很高，大于0.9864（阈值再调整），则直接使用结果，可处理一些常识固定搭配问题
    如：国家名[乌]克兰、[伊]拉克等
    :param sample_emoji: 测试集单条含emoji的文本
    :param fill_mask: 加载预训练参数的bert掩码填词模型
    :return: 高置信度翻译结果或者0
    """
    emoji_info = emoji.emoji_list(sample_emoji)
    if len(emoji_info) == 1:
        sentence = sample_emoji[:emoji_info[0]['match_start']] + '[MASK]' + sample_emoji[emoji_info[0]['match_end']:]
        out = fill_mask(sentence)[0]
        if out['score'] > 0.9864:
            # 去除不可能是emoji翻译的干扰翻译项
            if out['token_str'] in [',', '.', '，', '。', '！', '？', '你', '我', '他', '它', '她', '的', '人', '吧']:
                pass
            else:
                if out['token_str'] in ['了', '有', '样', '不', '个', '真', '因', '此', '谁', '来', '一', '如', '多']:
                    pass
                else:
                    sentence = sample_emoji[:emoji_info[0]['match_start']] + out['token_str'] + \
                               sample_emoji[emoji_info[0]['match_end']:]
                    return sentence
    return 0


def generate_sentence_ppl(sentence, model, tokenizer):
    """
    基于预训练参数计算句子的困惑度
    :param sentence: 待检测困惑度的句子
    :param model: bert模型
    :param tokenizer: bert的分词器
    :return: 困惑度分数
    """
    logging.set_verbosity_warning()
    with torch.no_grad():
        model.eval()

        tokenize_input = tokenizer.tokenize(sentence)
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
        sen_len = len(tokenize_input)
        sentence_loss = 0.

        for i, word in enumerate(tokenize_input):
            # add mask to i-th character of the sentence
            tokenize_input[i] = '[MASK]'
            mask_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])

            output = model(mask_input)

            prediction_scores = output[0]
            softmax = nn.Softmax(dim=0)
            ps = softmax(prediction_scores[0, i]).log()
            word_loss = ps[tensor_input[0, i]]
            sentence_loss += word_loss.item()

            tokenize_input[i] = word
        ppl = np.exp(-sentence_loss / sen_len)

        return ppl


def creat_fill_mask(log=True):
    """
    直接加载预训练的模型参数，不需要继续训练
    :return: 可用于掩码填词的模型
    """
    if log:
        print('--------------------generate_fill_mask--------------------')
    logging.set_verbosity_warning()
    logging.set_verbosity_error()
    model_path = "D:/dataset_all/bert_chinese/bert_base_chinese"
    fill_mask = pipeline(
        "fill-mask",
        model=model_path,
        tokenizer=model_path,
    )

    return fill_mask


def bert_handle_ride(sample_emoji, sentence, model, tokenizer):
    """
    使用bert单句困惑度判断特定emoji前是否需要加内容
    :param sample_emoji: 测试集样本
    :param sentence: 前面翻译好的句子
    :param model: bert模型
    :param tokenizer: bert分词器
    :return: 命中则返回翻译好的句子，否则返回原句子
    """

    emoji_info = emoji.emoji_list(sample_emoji)
    if emoji_info[0]['emoji'] == '🏍' or emoji_info[0]['emoji'] == '🚲':
        start_p = emoji_info[0]['match_start']
        best_sentence = sentence
        current_sentence = best_sentence[:start_p] + '骑' + best_sentence[emoji_info[0]['match_start']:]
        score0 = generate_sentence_ppl(best_sentence, model, tokenizer)
        score1 = generate_sentence_ppl(current_sentence, model, tokenizer)
        if score0 > score1:
            return current_sentence
        else:
            return sentence
    else:
        return sentence


def bert_handle_verb(sample_emoji, sentence, model, tokenizer, skip_judge=0):
    """
    使用bert语言模型困惑度处理动词冗余问题
    :param skip_judge:
    :param sample_emoji: 测试集样本
    :param sentence: 翻译后的测试集样本
    :param model: bert模型
    :param tokenizer: bert分词器
    :return: 进一步处理的翻译或者原始翻译
    """
    if len(emoji.emoji_list(sample_emoji)) == 1:
        if emoji.emoji_list(sample_emoji)[0]['match_start'] != 0:
            if sample_emoji[emoji.emoji_list(sample_emoji)[0]['match_start'] - 1] in ['在', '去', '要'] or skip_judge:
                result = ''.join(list(difflib.Differ().compare(sample_emoji, sentence)))
                word = ''
                for i in range(len(result)):
                    if result[i] == '+':
                        word += result[i + 2]
                # need_add_verb.append([sample_emoji, word])
                if len(word) > 2 or skip_judge:
                    emoji_info = emoji.emoji_list(sample_emoji)

                    left = sample_emoji.split(emoji_info[0]['emoji'])[0]
                    right = sample_emoji.split(emoji_info[0]['emoji'])[1]
                    sentence0 = left + word + right
                    sentence1 = left + word[:-1] + right
                    score0 = generate_sentence_ppl(sentence0, model, tokenizer)
                    score1 = generate_sentence_ppl(sentence1, model, tokenizer)
                    # 句子本身困惑度就很大就不处理了
                    if score0 < 500:
                        if score0 > score1:
                            return sentence1
                        if list(posseg.cut(word))[0].flag == 'n' and emoji.emoji_list(sample_emoji)[0]['emoji'] == '🛒':
                            return sentence1
    # 只不命中就返回原句子
    return sentence


def bert_handle_adj(sample_emoji, sentence, model, tokenizer, data_adj, mode='off-line'):
    """
    使用bert语言模型困惑度处理很加形容词问题
    :param sample_emoji: 测试集样本
    :param sentence: 翻译后的测试集样本
    :param model: bert模型
    :param tokenizer: bert分词器
    :param data_adj: 离线模式需要使用的替换词典
    :param mode:处理模式，off-lineon-line。在线会很慢，离线是将处理结果保存下来直接替换
    :return: 进一步处理的翻译结果或者原始翻译结果
    """
    if mode == 'off-line':
        for i in range(len(data_adj)):
            if data_adj[i][0].split('\t')[0] == sample_emoji:
                return data_adj[i][0].split('\t')[1]
        return sentence

    elif mode == 'on-line':
        emoji_candidate = generate_emoji_candidate()
        emoji_info = emoji.emoji_list(sample_emoji)
        if emoji_info[0]['match_start'] != 0:
            if sample_emoji[emoji_info[0]['match_start'] - 1] == '很':
                jieba_cut = list(jieba.cut(sentence))
                if '很' in jieba_cut:
                    if emoji_info[0]['emoji'] in emoji_candidate:
                        word_candidate = emoji_candidate[emoji_info[0]['emoji']]
                        # 限制最多考虑前五的答案
                        word_candidate = word_candidate[:5] if len(word_candidate) > 5 else word_candidate

                        left = sample_emoji.split(emoji_info[0]['emoji'])[0]
                        right = sample_emoji.split(emoji_info[0]['emoji'])[1]
                        score_list = []
                        sentence_list = []
                        for i in range(len(word_candidate)):
                            sentence = left + word_candidate[i] + right
                            sentence_list.append(sentence)
                            score_list.append(generate_sentence_ppl(sentence, model, tokenizer))

                        if min(score_list) < 21:
                            return sentence_list[score_list.index(min(score_list))]
        return sentence
    else:
        print('error.mode:off-line|on-line')
        return 0


def bert_choose_word(sample_emoji, model, tokenizer, emoji_candidate):
    emoji_info = emoji.emoji_list(sample_emoji)
    if len(emoji_info) == 1:
        if emoji_info[0]['emoji'] in emoji_candidate:
            word_candidate = emoji_candidate[emoji_info[0]['emoji']]
            if '' in word_candidate:
                word_candidate.remove('')
            if 30 < len(word_candidate) < 50:
                word_candidate = word_candidate[:10]
            elif 3 < len(word_candidate) <= 30 or len(word_candidate) >= 50:
                word_candidate = word_candidate[:3]
            else:
                pass
            score = []
            left = sample_emoji.split(emoji_info[0]['emoji'])[0]
            right = sample_emoji.split(emoji_info[0]['emoji'])[1]
            for i in range(len(word_candidate)):
                sentence = left + word_candidate[i] + right
                score.append(generate_sentence_ppl(sentence, model, tokenizer))
            return left + word_candidate[score.index(min(score))] + right
    return sample_emoji


def bert_handle_fixed_verb(sample_emoji, sentence, model, tokenizer, _id):
    word_candidate = ['拉', '看', '爬']
    emoji_info = emoji.emoji_list(sample_emoji)
    if len(emoji_info) == 1:
        if _id not in [10378, 10535, 10666, 10765, 10801, 11403]:
            if emoji_info[0]['emoji'] in ['📕', '⛰', '💩']:
                start_p = emoji_info[0]['match_start']
                score0 = generate_sentence_ppl(sentence, model, tokenizer)
                score = []
                best_sentence = sentence
                for i in range(len(word_candidate)):
                    current_sentence = best_sentence[:start_p] + word_candidate[i] + best_sentence[start_p:]
                    score.append(generate_sentence_ppl(current_sentence, model, tokenizer))
                if min(score) < score0:
                    return best_sentence[:start_p] + word_candidate[score.index(min(score))] + best_sentence[start_p:]
    return sentence


if __name__ == '__main__':
    model, tokenizer = initialize_bert()
    print(bert_handle_fixed_verb('周日一起去⛰', '周日一起去山', model, tokenizer))