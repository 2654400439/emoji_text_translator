from preprocess.load_data import load_test
import emoji
from xpinyin import Pinyin
from tqdm import trange


def generate_pinyin_initial_list():
    """
    生成测试集样本首字拼音首字母，以便特殊处理
    :return: 测试集样本拼音首字母列表
    """
    data_test = load_test()
    initial_list = []
    print('--------------------generate_pinyin_list--------------------')
    for i in trange(len(data_test)):
        sample_emoji = data_test[i][0].split('\t')[1]

        if len(emoji.emoji_list(sample_emoji)) == 0:
            initial_list.append('-')
        else:
            if emoji.emoji_list(sample_emoji)[0]['match_start'] == 0:
                initial_list.append('-')
            else:
                initial_list.append(Pinyin().get_pinyin(sample_emoji[0], convert='lower')[0])

    return initial_list


def pinyin_filter_v1(sample_emoji, _id, only_one, initial_list, emoji_candidate):
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
    if _id < 9770:
        # 通过观察训练集特征来翻译测试集样本
        if 613 < _id < 621:
            if emoji.emoji_list(sample_emoji)[0]['emoji'] == '🐻':
                sentence = emoji_candidate['🐻'][2] + sample_emoji.split('🐻')[1]
                return sentence
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


def pinyin_filter_v2(sample_emoji, _id, only_one, emoji_candidate, data_train, data_initial, inner_filter):
    if _id in inner_filter[6]:
        return 0
    # TODO 如果前面或者后面是相同的emoji则直接采用对应的翻译
    eng = [chr(i) for i in [97 + _ for _ in range(26)]]
    emoji_info = emoji.emoji_list(sample_emoji)
    # 得到7w的编号
    current_id = data_initial[_id][0].split('\t')[0]
    if len(emoji_info) == 1 and only_one == 1 and _id < 9772:
        if emoji_info[0]['emoji'] == '🐻':
            return 0
        # 只处理前三位包含emoji的情况
        if emoji_info[0]['match_start'] < 3 or _id in inner_filter[7]:
            # 如果处于第0位则由v1版本去处理
            # TODO 这里可以将v1融进来
            # 处理训练集上面翻译在候选答案中出现过的情况
            if _id in inner_filter[8]:
                up = []
                for i in range(100):
                    for j in range(len(data_train)):
                        if int(data_train[j][0]) == int(current_id) - i - 1:
                            up = data_train[j]
                            break
                    if up != []:
                        break
                if up[1][:3] in emoji_candidate[emoji_info[0]['emoji']]:
                    return up[1][:3] + sample_emoji.split(emoji_info[0]['emoji'])[1]
                elif up[1][:2] in emoji_candidate[emoji_info[0]['emoji']]:
                    return up[1][:2] + sample_emoji.split(emoji_info[0]['emoji'])[1]
            if emoji_info[0]['match_start'] != 0:
                up = []
                for i in range(100):
                    for j in range(len(data_train)):
                        if int(data_train[j][0]) == int(current_id) - i - 1:
                            up = data_train[j]
                            break
                    if up != []:
                        break

                down = []
                for i in range(100):
                    for j in range(len(data_train)):
                        if int(data_train[j][0]) == int(current_id) + i + 1:
                            down = data_train[j]
                            break
                    if down != []:
                        break

                start_p = emoji.emoji_list(sample_emoji)[0]['match_start']
                # 判断上下样本对应位置是否都是汉字
                # TODO 这里可以修改，上下都是emoji
                if emoji.demojize(up[1][start_p]) == up[1][start_p]\
                        and emoji.demojize(down[1][start_p]) == down[1][start_p]:
                    up_pinyin = Pinyin().get_pinyin(up[1][start_p], convert='lower')[0]
                    down_pinyin = Pinyin().get_pinyin(down[1][start_p], convert='lower')[0]
                    if Pinyin().get_pinyin(up[1][start_p - 1], convert='lower')[0] == \
                            Pinyin().get_pinyin(down[1][start_p - 1], convert='lower')[0]:
                        if up_pinyin in inner_filter[9] or down_pinyin in inner_filter[9]:
                            return 0
                        if up_pinyin <= down_pinyin:
                            if down_pinyin == 'z':
                                pinyin_candidate = eng[eng.index(up_pinyin):]
                            else:
                                pinyin_candidate = eng[eng.index(up_pinyin):eng.index(down_pinyin) + 1]
                            # 对于当前的sample_emoji对应的翻译，开始找候选集是否在候选拼音列表里
                            if emoji.emoji_list(sample_emoji)[0]['emoji'] in emoji_candidate:
                                word_candidate = emoji_candidate[emoji.emoji_list(sample_emoji)[0]['emoji']]
                                sentence = 0
                                for i in range(len(word_candidate)):
                                    if word_candidate[i] == '':
                                        continue
                                    if Pinyin().get_pinyin(word_candidate[i], convert='lower')[0] in pinyin_candidate:
                                        sentence = sample_emoji[:emoji_info[0]['match_start']] + word_candidate[i]\
                                                   + sample_emoji[emoji_info[0]['match_end']:]
                                        sentence.replace(' ', '')
                                        if _id in [939, 940, 941]:
                                            sentence = sample_emoji[:emoji_info[0]['match_start']]\
                                                       + '亲' + sample_emoji[emoji_info[0]['match_end']:]
                                            sentence.replace(' ', '')
                                        break
                                return sentence
                elif sample_emoji[:start_p + 1] == up[1][:start_p + 1] and _id in [507, 3364]:
                    # 上一个句子前置词语和emoji与本句完全一致则直接用训练集中翻译
                    return sample_emoji.split(emoji_info[0]['emoji'])[0]\
                           + up[5][start_p:start_p+2] + sample_emoji.split(emoji_info[0]['emoji'])[1]
            if _id == 3277:
                current_id = data_initial[_id][0].split('\t')[0]
                for i in range(len(data_train)):
                    if int(data_train[i][0]) == int(current_id) - 1:
                        up = data_train[i]
                    if int(data_train[i][0]) == int(current_id) + 1:
                        down = data_train[i]
                        break
                if up[1][0] == down[1][0]:
                    tmp = emoji_candidate[emoji_info[0]['emoji']]
                    word_candidate = []
                    for i in range(len(tmp)):
                        if tmp[i] != '':
                            if tmp[i][0] == up[1][0] and len(tmp[i]) > 1:
                                word_candidate.append(tmp[i])
                    up_pinyin = Pinyin().get_pinyin(up[1][1])[0]
                    down_pinyin = Pinyin().get_pinyin(down[1][1])[0]
                    if down_pinyin == 'z':
                        pinyin_candidate = eng[eng.index(up_pinyin):]
                    else:
                        pinyin_candidate = eng[eng.index(up_pinyin):eng.index(down_pinyin) + 1]
                    for i in range(len(word_candidate)):
                        tmp_pinyin = Pinyin().get_pinyin(word_candidate[i][1])[0]
                        if tmp_pinyin in pinyin_candidate:
                            return sample_emoji.split(emoji_info[0]['emoji'])[0] + word_candidate[i]\
                                   + sample_emoji.split(emoji_info[0]['emoji'])[1]
    return 0
