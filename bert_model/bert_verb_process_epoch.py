
import os
import json
from transformers import pipeline
import emoji as EMOJI

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

def get_bert_fill_mask_model(model_path):

    fill_mask_model = pipeline(
        "fill-mask",
        model=model_path,
        tokenizer=model_path,
    )
    return fill_mask_model

def process_bert_iter(fill_mask_model, verbs_list, sen_emoji, sen_transed):

    sen_emoji = sen_emoji.replace(" ", "")
    sen_transed = sen_transed.replace(" ", "")

    emojis = [ item['emoji'] for item in EMOJI.emoji_list(sen_emoji) ]
    if(len(emojis) >= 2 or len(emojis) == 0):  # 无法处理多个表情的情况
        return sen_transed

    emoji = emojis[0]
    first_part, second_part = sen_emoji.split(emoji)
    if(len(first_part) == 0): # 表情在头部不必处理
        return sen_transed

    test_line = first_part + '[MASK]' + sen_transed[len(first_part):]
    scores = fill_mask_model(test_line)
    score = scores[0]
    if(score['score'] > 0.85 and score['token_str'] in verbs_list and score['token_str'] != sen_transed[len(first_part)] and score['token_str'] != first_part[-1]):
        test_line = test_line.replace("[MASK]", score['token_str'])
        return test_line
    else:
        return sen_transed

def main():

    fill_mask_model = get_bert_fill_mask_model("./bert_base_chinese")

    with open("chinese_verb.txt", 'r', encoding='utf-8') as f:
        verbs_list = [line.strip() for line in f.readlines()]

    ############################### 测试

    sen_emoji = "周末一起🎵" # 原始测试集需要加载作为输入1，用于定位 MASK 的位置
    line = "周末一起音乐" # 经过 ngram_1.3_bert_5 + fixed 之后的被翻译的样本，作为输入2

    #### INPUT
    # sen_emoji: 原始测试集样本
    # line: 前面处理后翻译的样本
    #### Output
    # 周末一起听音乐
    res = process_bert_iter(fill_mask_model, verbs_list, sen_emoji, line)
    print(res) # 周末一起听音乐


if __name__ == '__main__':
    main()