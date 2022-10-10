import csv
import numpy as np
import os
import json


def load_train():
    """
    加载训练集数据
    :return: np格式训练集数据
    """
    file_path = "D:/python_project/emoji_text_translator/dataset/emoji7w.csv"
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        data = [row for row in reader]

    data = np.array(data)
    data = data[1:]

    return data


def load_test():
    """
    加载测试集数据
    :return: np格式测试集数据
    """
    file_path = "D:/python_project/emoji_text_translator/dataset/emoji7w-test_data.csv"
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        data = [row for row in reader]

    data = np.array(data)
    data = data[1:]

    return data


def load_emoji2chinese():
    """
    加载emoji2chinese文件中数据
    :return: 数组格式的未处理emojiall网站数据
    """
    file_path = "D:/python_project/emoji_text_translator/dataset/emoji2chinese.csv"
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        data = [row for row in reader]

    data = np.array(data)

    return data


def generate_emoji2chinese_dict():
    """
    将emoji2chinese文件中提取到的数据转化成字典格式，key：emoji   value：对应翻译
    :return: emoji2chinese字典数据
    """
    data_chinese = load_emoji2chinese()

    for i in range(len(data_chinese)):
        data_chinese[i][1] = data_chinese[i][1].split(':')[0]
        data_chinese[i][1] = data_chinese[i][1].split('：')[0]
    data_chinese_dict = dict(zip(data_chinese[:, 0], data_chinese[:, 1]))

    return data_chinese_dict


def load_emoji_4_columns():
    """
    加载emoji_4_columns文件数据
    :return: 原data_emoji
    """
    file_path = "D:/python_project/emoji_text_translator/dataset/emoji_dict_4_columns.csv"
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        data = [row for row in reader]

    data = np.array(data)

    return data


def generate_emoji_candidate():
    """
    根据训练集数据，生成训练集中出现过的emoji，对应候选翻译（按出现次数从大到小排序）
    :return: 词典类型的emoji对应翻译（按出现次数排序）
    """
    data_emoji = load_emoji_4_columns()

    emoji_candidate = {}
    for i in range(len(data_emoji)):
        if data_emoji[i][1][1:].find(',') == -1:
            emoji_candidate[data_emoji[i][0]] = [data_emoji[i][1][2:-2]]
        else:
            tmp = data_emoji[i][1][1:].replace(' ', '').split(',')
            for j in range(len(tmp)):
                tmp[j] = tmp[j][1:-1]
            emoji_candidate[data_emoji[i][0]] = tmp

    for i in range(len(list(emoji_candidate.keys()))):
        if len(emoji_candidate[list(emoji_candidate.keys())[i]]) > 1:
            emoji_candidate[list(emoji_candidate.keys())[i]][-1] = emoji_candidate[list(emoji_candidate.keys())[i]][-1][
                                                                   :-1]

    return emoji_candidate


def generate_emoji_dict():
    """
    根据训练集数据，生成单emoji对应出现次数最多的翻译
    :return: 词典类型，emoji与其对应出现次数最多翻译
    """
    data_emoji = load_emoji_4_columns()

    data_emoji_key = []
    for i in range(len(data_emoji)):
        data_emoji_key.append(data_emoji[i][0])

    emoji_dict = {}
    for i in range(len(data_emoji)):
        if data_emoji[i][1][1:].find(',') == -1:
            emoji_dict[data_emoji[i][0]] = data_emoji[i][1][2:-2]
        else:
            emoji_dict[data_emoji[i][0]] = data_emoji[i][1][1:].split(',')[0][1:-1]

    del emoji_dict['👌']

    return emoji_dict


def load_adj_data():
    file_path = "D:/python_project/emoji_text_translator/dataset/_adj_bert_change.csv"
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        data = [row for row in reader]

    data = np.array(data)

    return data


def load_data_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        data = [row for row in reader]

    data = np.array(data)

    return data


def json_load(file_path, log):
    if log:
        print(f"loading json data {file_path} ...")

    with open(file_path, 'r', encoding="utf-8") as f:
        json_data = json.load(f)

    if log:
        print(f"loading json data {file_path} Success")
    return json_data


def load_data_for_bert_verb(log=True):
    if os.path.exists("./dataset/verb_and_fixed_data/id2bertverbs.json"):
        id2bertverbs = json_load('./dataset/verb_and_fixed_data/id2bertverbs.json', log)
    else:
        id2bertverbs = {}

    with open("./dataset/verb_and_fixed_data/chinese_verb.txt", 'r', encoding='utf-8') as f:
        verbs_list = [line.strip() for line in f.readlines()]

    return id2bertverbs, verbs_list


if __name__ == '__main__':
    data_adj = load_adj_data()
    print(data_adj[0][0])