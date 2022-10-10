from preprocess.load_data import load_test, load_train, load_emoji_4_columns
import emoji
from collections import Counter
from tqdm import trange
from nltk.lm import Lidstone
from nltk.lm.preprocessing import padded_everygram_pipeline


def generate_handle_list():
    """
    生成需要使用ngram处理的emoji列表
    :return: emoji列表
    """
    # 加载需要的数据
    data_test = load_test()
    data_emoji = load_emoji_4_columns()
    data_emoji_key = []
    for i in range(len(data_emoji)):
        data_emoji_key.append(data_emoji[i][0])

    tmp = []
    for i in range(len(data_test)):
        sample_emoji = data_test[i][0].split('\t')[1]
        for j in range(len(emoji.emoji_list(sample_emoji))):
            tmp.append(emoji.emoji_list(sample_emoji)[j]['emoji'])
    # 通过在测试集中统计emoji出现情况，进行过滤
    Counter_tmp = Counter(tmp)
    tmp = dict(Counter_tmp)

    # 测试集里所有出现次数大于10的emoji
    handle_list = []
    for j in range(len(list(tmp.keys()))):
        if list(tmp.keys())[j] in data_emoji_key:
            if len(data_emoji[data_emoji_key.index(list(tmp.keys())[j])][1].split(',')) > 1:
                if tmp[list(tmp.keys())[j]] >= 10:
                    handle_list.append(list(tmp.keys())[j])

    # 过滤emoji候选翻译出现集中（不需要使用ngram）的情况
    handle_list.remove('😓')
    handle_list.remove('🔫')
    handle_list.remove('🍬')
    handle_list.remove('🎂')
    handle_list.remove('👓')
    handle_list.remove('🤡')
    handle_list.remove('🚉')
    handle_list.remove('🐩')
    handle_list.remove('🏥')
    handle_list.remove('🎧')
    handle_list.remove('🍚')
    handle_list.remove('👸🏻')
    handle_list.remove('🆚')
    handle_list.remove('🈵')
    handle_list.remove('🩲')
    handle_list.remove('💵')
    handle_list.remove('👋')
    handle_list.remove('👍')
    handle_list.remove('👌')
    handle_list.remove('🍅')
    handle_list.remove('👇')
    handle_list.remove('🌺')
    handle_list.remove('😣')
    handle_list.remove('🚶')
    handle_list.remove('🕳')
    handle_list.remove('😠')
    handle_list.remove('💧')
    handle_list.remove('🚄')
    # 加上🧄、📡、🎩、🙌🏻、🧪、😀、😉、🦄、🎓、🚭、👨‍🦲、🔋
    handle_list.append('🧄')
    handle_list.append('📡')
    handle_list.append('🎩')
    handle_list.append('🙌🏻')
    handle_list.append('🧪')
    handle_list.append('😀')
    handle_list.append('😉')
    handle_list.append('🦄')
    handle_list.append('🎓')
    handle_list.append('🚭')
    handle_list.append('🔋')
    handle_list.append('👨‍🦲')
    handle_list.append('🗡')
    handle_list.append('🎥')
    # 增加的尝试0926
    handle_list.append('👴🏻')

    return handle_list


def generate_ngram_data(special_emoji):
    """
    生成对于某个emoji训练ngram模型需要的数据
    :param special_emoji: 指定的某个emoji
    :return: 对应的训练集数据
    """
    data_train = load_train()

    data_ngram_data = []
    for i in range(len(data_train)):
        sample_emoji = data_train[i][2]
        if len(emoji.emoji_list(sample_emoji)) == 1:
            if emoji.emoji_list(sample_emoji)[0]['emoji'] == special_emoji:
                data_ngram_data.append([data_train[i][2], data_train[i][5]])

    # 针对只占一位的标准emoji生成供语言模型训练的数据集，除emoji部分为其余按字分割
    train = []
    for j in range(len(data_ngram_data)):
        sample_emoji = data_ngram_data[j][0]
        sample_new = sample_emoji.split(special_emoji)
        translate = data_ngram_data[j][1][emoji.emoji_list(sample_emoji)[0]['match_start']:data_ngram_data[j][1].find(sample_new[1])]
        left, right = [], []
        for i in range(len(sample_new[0])):
            left.append(sample_new[0][i])
        for i in range(len(sample_new[1])):
            right.append(sample_new[1][i])
        sentence = left + [translate] + right
        train.append(sentence)

    return train


def generate_ngram(special_emoji):
    """
    生成3gram语言模型，使用数据平滑的方法
    :param special_emoji: 特定的需要生成ngram的emoji
    :return: 该emoji的基于训练集的3gram语言模型
    """
    train_d = generate_ngram_data(special_emoji)

    ngram_order = 3

    train_data, vocab_data = padded_everygram_pipeline(ngram_order, train_d)

    lm = Lidstone(0.2, ngram_order)
    lm.fit(train_data, vocab_data)
    return lm


def generate_lm_list(handle_list):
    """
    生成handle_list中所有emoji的3gram语言模型
    :param handle_list: 待处理emoji列表
    :return: 语言模型列表
    """
    lm_list = []
    print('--------------------generate_lm_list--------------------')
    for i in trange(len(handle_list)):
        lm_list.append(generate_ngram(handle_list[i]))

    return lm_list


if __name__ == '__main__':
    handle_list = generate_handle_list()
    print(handle_list[0])
    lm_list = generate_lm_list(handle_list)
