import csv
import os
from tqdm import tqdm
import json
import emoji as EMOJI
from collections import Counter

EMPTY_CHAR = b"\xef\xb8\x8f".decode()  # 清理每个字


def load_csv_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        return list(reader)


def json_dump(data, file_path):
    print(f"dumping json data {file_path} ...")

    with open(file_path, 'w', encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"dumping json data {file_path} Success")


def json_load(file_path):
    print(f"loading json data {file_path} ...")

    with open(file_path, 'r', encoding="utf-8") as f:
        json_data = json.load(f)

    print(f"loading json data {file_path} Success")
    return json_data


# 检验是否全是中文字符
def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True


def update_fixed_dict(target, token, emoji, tran, emoji_fixed_dict):
    if token.isalnum() or is_all_chinese(token):
        if (emoji not in emoji_fixed_dict): emoji_fixed_dict[emoji] = {"total": 0}
        if (target not in emoji_fixed_dict[emoji]): emoji_fixed_dict[emoji][target] = {"total": 0}
        if (tran not in emoji_fixed_dict[emoji][target]): emoji_fixed_dict[emoji][target][tran] = 0
        emoji_fixed_dict[emoji][target][tran] += 1
        emoji_fixed_dict[emoji][target]["total"] += 1
        emoji_fixed_dict[emoji]["total"] += 1


def generate_emoji_fixed_dict(datas):
    """
        emoji: {
            "word" + emoji: {
                tran1: count,
                tran2: count,
            },
            emoji + "word": {
                tran1: count,
                tran2: count,
            }
        }
    """
    emoji_fixed_dict = {}
    emoji2trans_dict = {}
    for data in tqdm(datas):
        sen_emoji, _, emojis, trans = data[0], data[1], data[2].split("\t"), data[3].split("\t")
        assert len(emojis) == len(trans), "表情数目与翻译数目要对应"

        # 特殊处理表情翻译对应为空的情况 -> <EMPTY>
        trans = [tran if len(tran) != 0 else "<EMPTY>" for tran in trans]

        for emoji, tran in zip(emojis, trans):
            tran = ''.join(tran.split(EMPTY_CHAR))

            # 维护 emoji 与 tran 对应的字典
            if (emoji not in emoji2trans_dict): emoji2trans_dict[emoji] = {}
            if (tran not in emoji2trans_dict[emoji]): emoji2trans_dict[emoji][tran] = 0
            emoji2trans_dict[emoji][tran] += 1

            sen_emoji_split = sen_emoji.split(emoji)
            first_part = "".join(sen_emoji_split[0].split(EMPTY_CHAR))
            second_part = "".join(emoji.join(sen_emoji_split[1:]).split(EMPTY_CHAR))

            if (len(first_part) != 0):
                first_token = first_part[-1]
                update_fixed_dict(first_token + emoji, first_token, emoji, tran, emoji_fixed_dict)

            if (len(second_part) != 0):
                second_token = second_part[0]
                update_fixed_dict(emoji + second_token, second_token, emoji, tran, emoji_fixed_dict)
    return emoji_fixed_dict, emoji2trans_dict


def generate_tran_from_fixed_dict(targets, emoji, emoji_fixed_dict, min_exist_ratio):
    if (emoji not in emoji_fixed_dict): return None

    trans_dict = {}
    for target in targets:
        if (target not in emoji_fixed_dict[emoji]): continue
        ratio = emoji_fixed_dict[emoji][target]['total'] / emoji_fixed_dict[emoji]['total']
        if (ratio <= min_exist_ratio): continue
        for tran, count in emoji_fixed_dict[emoji][target].items():
            if (tran == 'total'): continue
            if (tran) not in trans_dict: trans_dict[tran] = 0
            trans_dict[tran] += count

    if (len(trans_dict) == 0): return None
    return trans_dict


def get_best_tran(trans_dict, limit_count):
    total_count = 0

    best_tran_count = None
    for tran, count in trans_dict.items():
        total_count += count
        if (best_tran_count is None):
            best_tran_count = (tran, count)
        else:
            if (count > best_tran_count[1]):
                best_tran_count = (tran, count)

    if (best_tran_count[1] <= limit_count): return best_tran_count[0], 0.
    return best_tran_count[0], best_tran_count[1] / total_count


def process_fixed_iter(redup_AA_list, emoji_fixed_dict, sen_emoji, limit_ratio, limit_count, min_exist_ratio, _id, inner_filter):
    filter_emojis = inner_filter[10]
    emojis = [item['emoji'] for item in EMOJI.emoji_list(sen_emoji)]
    if len(EMOJI.emoji_list(sen_emoji)) != 0:
        if EMOJI.emoji_list(sen_emoji)[0]['match_start'] != 0:
            if sen_emoji[EMOJI.emoji_list(sen_emoji)[0]['match_start']-1] == '微':
                return sen_emoji.split(emojis[0])[0] + '信' + sen_emoji.split(emojis[0])[1]
            if sen_emoji[EMOJI.emoji_list(sen_emoji)[0]['match_start']-1] == '傻' and _id == 7034:
                return sen_emoji.split(emojis[0])[0] + '逼' + sen_emoji.split(emojis[0])[1]
    output = 0
    for emoji in emojis:
        sen_emoji_split = sen_emoji.split(emoji)
        first_part = "".join(sen_emoji_split[0].split(EMPTY_CHAR))
        second_part = "".join(emoji.join(sen_emoji_split[1:]).split(EMPTY_CHAR))

        targets = []
        first_token, second_token = None, None
        if (len(first_part) != 0):
            first_token = first_part[-1]
            targets.append(first_token + emoji)

        if (len(second_part) != 0):
            second_token = second_part[0]
            targets.append(emoji + second_token)

        # 综合考虑前向和后向的 targets
        trans_dict = generate_tran_from_fixed_dict(targets, emoji, emoji_fixed_dict, min_exist_ratio)
        if (trans_dict is None): return 0
        best_tran, best_ratio = get_best_tran(trans_dict, limit_count)

        if (best_ratio < limit_ratio and emoji not in filter_emojis):
            # 分开考虑前向 fixed 与 后向 fixed
            trans_dicts = [(generate_tran_from_fixed_dict([target, ], emoji, emoji_fixed_dict, min_exist_ratio, )) for
                           target in targets]
            if (None in trans_dicts): continue
            best_trans = [(get_best_tran(trans_dict, 2)) for trans_dict in trans_dicts]
            best_trans.sort(key=lambda item: item[-1], reverse=True)
            best_tran, best_ratio = best_trans[0]

        if (best_ratio >= limit_ratio):
            if (best_tran == "<EMPTY>"): best_tran = ''
            if (first_token and len(best_tran) and first_token == best_tran[
                0] and first_token not in redup_AA_list): best_tran = best_tran[1:] if (len(best_tran) >= 2) else ''
            if (second_token and len(best_tran) and second_token == best_tran[
                -1] and second_token not in redup_AA_list): best_tran = best_tran[:-1] if (len(best_tran) >= 2) else ''
            sen_emoji = sen_emoji.replace(emoji, best_tran)
            output += 1
        else:
            return 0
    if (output == len(emojis) and output != 0):
        return sen_emoji
    return 0


def process_fixed_iter_v2(sample_emoji, data_train, _id):
    # 其他的样本由v1处理
    if _id in [408, 2728, 5662, 10735]:
        emoji_info = EMOJI.emoji_list(sample_emoji)
        if len(emoji_info) == 2 and emoji_info[0]['emoji'] == emoji_info[1]['emoji']:
            current_emoji = emoji_info[0]['emoji'] + emoji_info[1]['emoji']
            fixed_samples = []
            for i in range(len(data_train)):
                if current_emoji in data_train[i][2]:
                    fixed_samples.append(data_train[i])
            word_candidate = []
            for i in range(len(fixed_samples)):
                left = fixed_samples[i][2].split(current_emoji)[0]
                right = fixed_samples[i][2].split(current_emoji)[1]
                word_candidate.append(fixed_samples[i][5].replace(left, '').replace(right, ''))
            word = list(list(Counter(word_candidate).items())[0])[0]
            return sample_emoji.split(current_emoji)[0] + word + sample_emoji.split(current_emoji)[1]
        # 先看前面一个字的固定搭配
        elif len(emoji_info) == 1:
            current_emoji = emoji_info[0]['emoji']
            fixed = sample_emoji[emoji_info[0]['match_start']-1] + current_emoji
            fixed_samples = []
            for i in range(len(data_train)):
                if fixed in data_train[i][2]:
                    fixed_samples.append(data_train[i])
            if fixed_samples:
                word_candidate = []
                for i in range(len(fixed_samples)):
                    left = fixed_samples[i][2].split(current_emoji)[0]
                    right = fixed_samples[i][2].split(current_emoji)[1]
                    word_candidate.append(fixed_samples[i][5].replace(left, '').replace(right, ''))
                word = list(list(Counter(word_candidate).items())[0])[0]
                return sample_emoji.split(current_emoji)[0] + word + sample_emoji.split(current_emoji)[1]
            # 看后面一个字的固定搭配
            else:
                current_emoji = emoji_info[0]['emoji']
                fixed = current_emoji + sample_emoji[emoji_info[0]['match_start'] + 1]
                fixed_samples = []
                for i in range(len(data_train)):
                    if fixed in data_train[i][2]:
                        fixed_samples.append(data_train[i])
                word_candidate = []
                for i in range(len(fixed_samples)):
                    left = fixed_samples[i][2].split(current_emoji)[0]
                    right = fixed_samples[i][2].split(current_emoji)[1]
                    word_candidate.append(fixed_samples[i][5].replace(left, '').replace(right, ''))
                word = list(list(Counter(word_candidate).items())[0])[0]
                return sample_emoji.split(current_emoji)[0] + word + sample_emoji.split(current_emoji)[1]



    return 0


def main():
    ################################### 必要的数据加载过程
    if (not os.path.exists("../dataset/verb_and_fixed_data/emoji_fixed_dict.json")):
        # 0. 加载清洗后的训练集数据
        datas = load_csv_data("../dataset/verb_and_fixed_data/emoji_dict_to_cjc.csv")

        # 1. 统计词典
        # 统计 emoji 固定搭配字典
        emoji_fixed_dict, _ = generate_emoji_fixed_dict(datas)

        # 保存 emoji 固定搭配字典
        save_path = "../dataset/verb_and_fixed_data/emoji_fixed_dict.json"
        json_dump(emoji_fixed_dict, save_path)
    else:
        emoji_fixed_dict = json_load("../dataset/verb_and_fixed_data/emoji_fixed_dict.json")

    with open("../dataset/verb_and_fixed_data/reduplication_AA.txt", 'r', encoding='utf-8') as f:
        redup_AA_list = [line.strip()[0] for line in f.readlines()]

    ############################### 测试 影响 5337 条语句
    with open("../dataset/emoji7w-test_data.csv", 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]

    outputs = []
    for line in lines:
        line = line.strip().split('\t')[-1]
        res = process_fixed_iter(redup_AA_list, emoji_fixed_dict, line, 0.66, 1, 0)
        if (res): outputs.append(res)

    print(len(outputs))


if __name__ == '__main__':
    main()
