from preprocess.load_data import load_emoji_4_columns, generate_emoji_dict, generate_emoji_candidate
from standard_translator.handle_redundancy import detect_redundancy
import numpy as np
import emoji


def handle_sentence_with_special_emoji(sample_emoji, lm, emoji_special='🐻'):
    """
    输入sample_emoji，输出翻译后的句子, 只处理长度大于3且含有单emoji的句子
    :param sample_emoji: 测试集样本
    :param lm: 所有的3gram语言模型
    :param emoji_special: 特定的emoji用于选择语言模型
    :return: 使用3gram翻译的结果
    """
    data_emoji = load_emoji_4_columns()

    data_emoji_key = []
    for i in range(len(data_emoji)):
        data_emoji_key.append(data_emoji[i][0])

    candidate = data_emoji[data_emoji_key.index(emoji_special)][1][1:-1].split(', ')
    for i in range(len(candidate)):
        candidate[i] = candidate[i][1:-1]

    # 处理emoji在开头和在结尾的情况，如果在最开头和最结尾就只能计算两种
    start_p = emoji.emoji_list(sample_emoji)[0]['match_start']
    end_p = emoji.emoji_list(sample_emoji)[0]['match_end']
    if start_p == 0:
        # 需要加一个<s>
        left = ['<s>']
        right = [sample_emoji[end_p], sample_emoji[end_p + 1]]
        score1 = []
        for i in range(len(candidate)):
            context1 = (left[0], candidate[i])
            word1 = right[0]
            score1.append(lm.score(word1, context1))
        score2 = []
        for i in range(len(candidate)):
            context2 = (candidate[i], right[0])
            word2 = right[1]
            score2.append(lm.score(word2, context2))
        score = np.array(score1) + np.array(score2)
        if np.sort(score)[-1] / np.sort(score)[-2] > 1.3:
            sentence = detect_redundancy(sample_emoji, candidate[score.argmax()])
        else:
            sentence = detect_redundancy(sample_emoji, candidate[0])
        return sentence
    elif end_p == len(sample_emoji):
        # 需要加一个</s>
        left = [sample_emoji[start_p - 2], sample_emoji[start_p - 1]]
        right = ['</s>']
        score1 = []
        for i in range(len(candidate)):
            context1 = (left[0], left[1])
            word1 = candidate[i]
            score1.append(lm.score(word1, context1))
        score2 = []
        for i in range(len(candidate)):
            context2 = (left[1], candidate[i])
            word2 = right[0]
            score2.append(lm.score(word2, context2))
        score = np.array(score1) + np.array(score2)
        if np.sort(score)[-1] / np.sort(score)[-2] > 1.3:
            sentence = detect_redundancy(sample_emoji, candidate[score.argmax()])
        else:
            sentence = detect_redundancy(sample_emoji, candidate[0])
        return sentence
    elif start_p == 1:
        # 需要加一个<s>
        left = ['<s>', sample_emoji[start_p - 1]]
        right = [sample_emoji[end_p], sample_emoji[end_p + 1]]
        score1 = []
        for i in range(len(candidate)):
            context1 = (left[0], left[1])
            word1 = candidate[i]
            score1.append(lm.score(word1, context1))
        score2 = []
        for i in range(len(candidate)):
            context2 = (left[1], candidate[i])
            word2 = right[0]
            score2.append(lm.score(word2, context2))
        score3 = []
        for i in range(len(candidate)):
            context3 = (candidate[i], right[0])
            word3 = right[1]
            score3.append(lm.score(word3, context3))
        score = np.array(score1) + np.array(score2) + np.array(score3)
        if np.sort(score)[-1] / np.sort(score)[-2] > 1.3:
            sentence = detect_redundancy(sample_emoji, candidate[score.argmax()])
        else:
            sentence = detect_redundancy(sample_emoji, candidate[0])
        return sentence
    elif end_p == len(sample_emoji) - 1:
        # 需要加一个</s>
        left = [sample_emoji[start_p - 2], sample_emoji[start_p - 1]]
        right = [sample_emoji[end_p], '</s>']
        score1 = []
        for i in range(len(candidate)):
            context1 = (left[0], left[1])
            word1 = candidate[i]
            score1.append(lm.score(word1, context1))
        score2 = []
        for i in range(len(candidate)):
            context2 = (left[1], candidate[i])
            word2 = right[0]
            score2.append(lm.score(word2, context2))
        score3 = []
        for i in range(len(candidate)):
            context3 = (candidate[i], right[0])
            word3 = right[1]
            score3.append(lm.score(word3, context3))
        score = np.array(score1) + np.array(score2) + np.array(score3)
        if np.sort(score)[-1] / np.sort(score)[-2] > 1.3:
            #             sentence = sample_emoji[:start_p] + candidate[score.argmax()] + sample_emoji[end_p:]
            sentence = detect_redundancy(sample_emoji, candidate[score.argmax()])
        else:
            #             sentence = sample_emoji[:start_p] + candidate[0] + sample_emoji[end_p:]
            sentence = detect_redundancy(sample_emoji, candidate[0])
        return sentence
    else:
        # 标准处理过程
        left = [sample_emoji[start_p - 2], sample_emoji[start_p - 1]]
        right = [sample_emoji[end_p], sample_emoji[end_p + 1]]
        score1 = []
        for i in range(len(candidate)):
            context1 = (left[0], left[1])
            word1 = candidate[i]
            score1.append(lm.score(word1, context1))
        score2 = []
        for i in range(len(candidate)):
            context2 = (left[1], candidate[i])
            word2 = right[0]
            score2.append(lm.score(word2, context2))
        score3 = []
        for i in range(len(candidate)):
            context3 = (candidate[i], right[0])
            word3 = right[1]
            score3.append(lm.score(word3, context3))
        score = np.array(score1) + np.array(score2) + np.array(score3)
        if np.sort(score)[-1] / np.sort(score)[-2] > 1.3:
            #             sentence = sample_emoji[:start_p] + candidate[score.argmax()] + sample_emoji[end_p:]
            sentence = detect_redundancy(sample_emoji, candidate[score.argmax()])
        else:
            #             sentence = sample_emoji[:start_p] + candidate[0] + sample_emoji[end_p:]
            sentence = detect_redundancy(sample_emoji, candidate[0])
        return sentence


def handle_not_process(sample_emoji):
    emoji_candidate = generate_emoji_candidate()
    emoji_info = emoji.emoji_list(sample_emoji)
    if emoji_info[0]['match_start'] == 0:
        start_p = emoji_info[0]['match_start']
        end_p = emoji_info[0]['match_end']
        if end_p - start_p == len(sample_emoji):
            right_chr = ''
        else:
            right_chr = sample_emoji[end_p]
        # 处理emoji库出现的特殊错误
        if emoji_info[0]['emoji'] in emoji_candidate:
            tmp = emoji_candidate[emoji_info[0]['emoji']]
            tmp.remove('') if '' in tmp else tmp
            if tmp[0] == emoji_info[0]['emoji']:
                word = ''
            else:
                word = tmp[0]
        else:
            word = ''

        if right_chr == '🐸️'[1]:
            sentence = word + sample_emoji[emoji_info[0]['match_end'] + 1:]
        else:
            sentence = word + sample_emoji[emoji_info[0]['match_end']:]

    # 处理单emoji在结尾的情况
    elif emoji_info[0]['match_end'] == len(sample_emoji):
        start_p = emoji_info[0]['match_start']
        if emoji_info[0]['emoji'] in emoji_candidate:
            tmp = emoji_candidate[emoji_info[0]['emoji']]
            tmp.remove('') if '' in tmp else tmp
            if tmp[0] == emoji_info[0]['emoji']:
                word = ''
            else:
                word = tmp[0]
        else:
            word = ''
        sentence = sample_emoji[:emoji_info[0]['match_start']] + word

    # 处理一般情况，单emoji在句中
    else:
        start_p = emoji_info[0]['match_start']
        end_p = emoji_info[0]['match_end']
        left_chr = sample_emoji[start_p - 1]
        right_chr = sample_emoji[end_p]
        if emoji_info[0]['emoji'] in emoji_candidate:
            tmp = emoji_candidate[emoji_info[0]['emoji']]
            tmp.remove('') if '' in tmp else tmp
            if tmp[0] == emoji_info[0]['emoji']:
                word = ''
                return sample_emoji.split(tmp[0])[0] + sample_emoji.split(tmp[0])[1]
            else:
                word = tmp[0]
        else:
            word = ''

        if right_chr == '🐸️'[1]:
            if end_p + 1 == len(sample_emoji):
                sentence = sample_emoji[:emoji_info[0]['match_start']] + word
            else:
                sentence = sample_emoji[:emoji_info[0]['match_start']] \
                           + word + sample_emoji[emoji_info[0]['match_end'] + 1:]
        else:
            sentence = detect_redundancy(sample_emoji, word)

    return sentence
