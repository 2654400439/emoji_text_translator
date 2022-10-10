from ngram_model.ngram_utils import handle_sentence_with_special_emoji
from standard_translator.handle_redundancy import detect_redundancy
from standard_translator.handle_emoji2chinese import emoji2chinese_dict
from standard_translator.handle_unknown import emoji2english2chinese
from standard_translator.handle_same import handle_same_emoji
from preprocess.load_data import generate_emoji_dict, generate_emoji2chinese_dict, generate_emoji_candidate
from ngram_model.ngram import generate_handle_list, generate_lm_list
from preprocess.load_pickle_data import load_lm_list
from ngram_model.ngram_utils import handle_not_process
import emoji
import jieba
from xpinyin import Pinyin


def translator_based_on_pinyin(sample_emoji, _id, only_one, initial_list, emoji_candidate):
    """
    开头拼音检测(测试集小bug)
    在候选答案前3里选拼音一致的，有结果直接返回，没有结果就继续下面处理
    只有该分句出现在整句开头时才做处理
    :param sample_emoji: 测试集中包含emoji的句子
    :param _id: 训练集中样本实际序号
    :param only_one: 表示该分句是否处在全句开头
    :param initial_list: 测试集所有样本开头汉字拼音首字母
    :param emoji_candidate: 单emoji对应候选所有翻译
    :return: 基于拼音规则的翻译后的句子或者0
    """
    if 0 < _id < 9770:
        emoji_info = emoji.emoji_list(sample_emoji)

        if only_one == 1:
            if emoji_info[0]['match_start'] == 0:
                up = initial_list[_id - 1]
                if up == '-':
                    for i in range(6):
                        if initial_list[_id - (i + 2)] != '-':
                            up = initial_list[_id - (i + 2)]
                            break
                down = initial_list[_id + 1]
                if down == '-':
                    for i in range(6):
                        if initial_list[_id + (i + 2)] != '-':
                            down = initial_list[_id + (i + 2)]
                            break

                if up != '-' and down != '-':
                    if up == down:
                        current_emoji = emoji.emoji_list(sample_emoji)[0]['emoji']
                        if current_emoji in emoji_candidate:
                            for j in range(len(emoji_candidate[current_emoji])):
                                if emoji_candidate[current_emoji][j] != '':
                                    if Pinyin().get_pinyin(emoji_candidate[current_emoji][j][0], convert='lower')[0] \
                                            == up:
                                        return emoji_candidate[current_emoji][j] + sample_emoji[
                                                                                   emoji_info[0]['match_end']:]
                    else:
                        pass
    return 0


# 处理测试集单句函数
# 输入是待翻译句子和候选答案（使用新生成的emoji_all做候选答案）
def emoji2sentence_one(sample_emoji, emoji_dict, data_chinese_dict, handle_list, lm_list, emoji_candidate, only_one=0, initial_list=0, _id=0):
    emoji_info = emoji.emoji_list(sample_emoji)

    # 基于拼音规则过滤一遍，翻译出来就直接返回结果，否则就使用正常步骤继续处理
    sentence = translator_based_on_pinyin(sample_emoji, _id, only_one, initial_list, emoji_candidate)
    if sentence == 0:
        pass
    else:
        return sentence

    # 对于含有生成了3gram语言模型的emoji句子，使用3gram语言模型处理
    if len(sample_emoji) - (emoji_info[0]['match_end'] - emoji_info[0]['match_start']) > 3:
        if emoji_info[0]['emoji'] in handle_list:
            return handle_sentence_with_special_emoji(sample_emoji, lm_list[handle_list.index(emoji_info[0]['emoji'])], emoji_info[0]['emoji'])

    # 标准处理过程  如果emoji在训练集中出现过则使用出现次数最多的翻译直接替换；如果没出现过则检测emojiall数据中是否有；还没有的话就英文翻译处理
    # 处理单emoji在开头的情况
    if emoji_info[0]['match_start'] == 0:
        start_p = emoji_info[0]['match_start']
        end_p = emoji_info[0]['match_end']
        if end_p - start_p == len(sample_emoji):
            right_chr = ''
        else:
            right_chr = sample_emoji[end_p]
        # 处理emoji库出现的特殊错误
        if emoji_info[0]['emoji'] in emoji_dict:
            word = emoji_dict[emoji_info[0]['emoji']]
            if word == '' and emoji_info[0]['emoji'] in data_chinese_dict:
                word = emoji2chinese_dict(sample_emoji, data_chinese_dict, emoji_dict)
                if len(word) > 2:
                    word = ''
        elif emoji_info[0]['emoji'] in data_chinese_dict:
            #             word = data_chinese_dict[emoji_info[0]['emoji']]
            word = emoji2chinese_dict(sample_emoji, data_chinese_dict, emoji_dict)
        else:
            if _id == 1985:
                word = ''
            else:
                word = emoji2english2chinese(sample_emoji)

        if right_chr == '🐸️'[1]:
            sentence = word + sample_emoji[emoji_info[0]['match_end' ] +1:]
        else:
            sentence = word + sample_emoji[emoji_info[0]['match_end']:]

    # 处理单emoji在结尾的情况
    elif emoji_info[0]['match_end'] == len(sample_emoji):
        start_p = emoji_info[0]['match_start']
        if emoji_info[0]['emoji'] in emoji_dict:
            word = emoji_dict[emoji_info[0]['emoji']]
            if word == '' and emoji_info[0]['emoji'] in data_chinese_dict:
                word = emoji2chinese_dict(sample_emoji, data_chinese_dict, emoji_dict)
                if len(word) > 2:
                    word = ''
        elif emoji_info[0]['emoji'] in data_chinese_dict:
            word = emoji2chinese_dict(sample_emoji, data_chinese_dict, emoji_dict)
        else:
            word = emoji2english2chinese(sample_emoji)
        sentence = sample_emoji[:emoji_info[0]['match_start']] + word

    # 处理一般情况，单emoji在句中
    else:
        start_p = emoji_info[0]['match_start']
        end_p = emoji_info[0]['match_end']
        left_chr = sample_emoji[start_p - 1]
        right_chr = sample_emoji[end_p]
        if emoji_info[0]['emoji'] in emoji_dict:
            word = emoji_dict[emoji_info[0]['emoji']]
            # 不可见特殊字符检测
            if len(word) > 2 and word[2] == '2⃣️'[2]:
                word = word[0]
            if word == '' and emoji_info[0]['emoji'] in data_chinese_dict:
                word = emoji2chinese_dict(sample_emoji, data_chinese_dict, emoji_dict)
                if len(word) > 2:
                    word = ''
                if _id == 10777:
                    return sample_emoji.split(emoji_info[0]['emoji'])[0] + sample_emoji.split(emoji_info[0]['emoji'])[1]
        elif emoji_info[0]['emoji'] in data_chinese_dict:
            word = emoji2chinese_dict(sample_emoji, data_chinese_dict, emoji_dict)
        else:
            if _id == 2794:
                return sample_emoji.split(emoji_info[0]['emoji'])[0] + sample_emoji.split(emoji_info[0]['emoji'])[1]
            else:
                word = emoji2english2chinese(sample_emoji)

        if right_chr == '🐸️'[1]:
            if end_p + 1 == len(sample_emoji):
                sentence = sample_emoji[:emoji_info[0]['match_start']] + word
            else:
                sentence = sample_emoji[:emoji_info[0]['match_start']]\
                           + word + sample_emoji[emoji_info[0]['match_end' ] +1:]
        else:
            sentence = detect_redundancy(sample_emoji, word)

    return sentence


def emoji2sentence_one_handle_empty(sample_emoji, emoji_dict, data_chinese_dict, emoji_candidate, handle_list, lm_list, only_one, initial_list=0, _id=0):
    """
    在emoji2sentence_one的基础上检测当前emoji是否应该翻译为空
    :param 参考emoji2sentence_one参数定义
    :return: 返回emoji翻译为空的句子或者调用emoji2sentence_one继续处理
    """
    emoji_current = emoji.emoji_list(sample_emoji)[0]['emoji']
    if emoji_current in emoji_candidate:
        if emoji_current == '🈵' or emoji_current == '💋' or emoji_current == '🐑' or emoji_current == '😊':
            pass
        else:
            if emoji.emoji_list(sample_emoji)[0]['match_start'] != 0:
                left = list(jieba.cut(sample_emoji[:emoji.emoji_list(sample_emoji)[0]['match_start']]))[-1]
                if len(emoji_candidate[emoji_current]) > 5:
                    if left in emoji_candidate[emoji_current][:int(len(emoji_candidate[emoji_current]) / 5)]:
                        return sample_emoji[:emoji.emoji_list(sample_emoji)[0]['match_start']]\
                               + sample_emoji[emoji.emoji_list(sample_emoji)[0]['match_end']:]
                else:
                    if left in emoji_candidate[emoji_current][:len(emoji_candidate[emoji_current])]:
                        return sample_emoji[:emoji.emoji_list(sample_emoji)[0]['match_start']]\
                               + sample_emoji[emoji.emoji_list(sample_emoji)[0]['match_end']:]
                    if left == emoji_dict[emoji.emoji_list(sample_emoji)[0]['emoji']][:2]:
                        if _id != 2023:
                            return emoji_dict[emoji.emoji_list(sample_emoji)[0]['emoji']]\
                               + sample_emoji.split(emoji.emoji_list(sample_emoji)[0]['emoji'])[1]
    return emoji2sentence_one(sample_emoji, emoji_dict, data_chinese_dict, handle_list, lm_list, emoji_candidate,
                              only_one, initial_list, _id)


def general_process(sample_emoji, log=True):
    # 加载emoji_dict即训练集中出现过的emoji对应出现次数最多的翻译
    emoji_dict = generate_emoji_dict()
    # 加载data_chinese_dict，emojiall中的数据
    data_chinese_dict = generate_emoji2chinese_dict()
    # 加载emoji_candidate，训练集中出现emoji及其所有候选翻译
    emoji_candidate = generate_emoji_candidate()
    # 生成3gram语言模型
    try:
        lm_list = load_lm_list('./dataset/lm_list.data')
        handle_list = generate_handle_list()
        if log:
            print('--------------------generate_lm_list--------------------')
    except FileNotFoundError as e:
        if log:
            print('未找到指定pickle文件：', e)
            print('开始直接生成')
        # 加载待使用3gram语言模型处理的emoji列表
        handle_list = generate_handle_list()
        # 生成需要使用的3gram语言模型
        lm_list = generate_lm_list(handle_list)

    # 处理连续两个相同emoji的情况
    if len(emoji.emoji_list(sample_emoji)) == 2:
        sample_emoji = handle_same_emoji(sample_emoji)

    if len(emoji.emoji_list(sample_emoji)) == 1:
        sentence = emoji2sentence_one_handle_empty(sample_emoji, emoji_dict, data_chinese_dict, emoji_candidate,
                                                   handle_list, lm_list, 1)
        # 对于漏掉的样本继续处理
        if len(emoji.emoji_list(sentence)) == 1:
            sentence = handle_not_process(sentence)
    # 处理一条样本有多个emoji的情况，将多个emoji部分拆分成只包含一个emoji的子句
    else:
        sentence = ''
        for j in range(len(emoji.emoji_list(sample_emoji))):
            if j == 0:
                sample_emoji_current = sample_emoji[:emoji.emoji_list(sample_emoji)[j + 1]['match_start']]
                sample_emoji_current = emoji2sentence_one_handle_empty(sample_emoji_current, emoji_dict,
                                                                       data_chinese_dict, emoji_candidate,
                                                                       handle_list, lm_list, 1)
            elif j == len(emoji.emoji_list(sample_emoji)) - 1:
                sample_emoji_current = sample_emoji[emoji.emoji_list(sample_emoji)[j]['match_start']:]
                sample_emoji_current = emoji2sentence_one_handle_empty(sample_emoji_current, emoji_dict,
                                                                       data_chinese_dict, emoji_candidate,
                                                                       handle_list, lm_list, 0)
            else:
                sample_emoji_current = sample_emoji[emoji.emoji_list(sample_emoji)[j]['match_start']:
                                                    emoji.emoji_list(sample_emoji)[j + 1]['match_start']]
                sample_emoji_current = emoji2sentence_one_handle_empty(sample_emoji_current, emoji_dict,
                                                                       data_chinese_dict, emoji_candidate,
                                                                       handle_list, lm_list, 0)
            sentence += sample_emoji_current
        # 对于漏掉的样本继续处理
        if len(emoji.emoji_list(sentence)) == 1:
            sentence = handle_not_process(sentence)
    return sentence
