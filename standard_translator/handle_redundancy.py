import emoji


def detect_redundancy(sample_emoji, word):
    """
    检测翻译左右是否有一模一样的字，以便去除冗余
    :param sample_emoji: 测试集样本
    :param word: 翻译结果
    :return:
    """
    if word == '':
        return sample_emoji
        # emoji_info = emoji.emoji_list(sample_emoji)
        # return sample_emoji.split(emoji_info[0]['emoji'])[0] + sample_emoji.split(emoji_info[0]['emoji'])[1]

    emoji_info = emoji.emoji_list(sample_emoji)
    # 如果emoji不在开头就对左边一个字进行检查
    if emoji_info[0]['match_start'] != 0:
        if word[0] == sample_emoji[emoji_info[0]['match_start'] - 1]:
            sample_emoji = sample_emoji[:emoji_info[0]['match_start']][:-1] + emoji_info[0]['emoji']\
                           + sample_emoji[emoji_info[0]['match_end']:]
        else:
            pass

    emoji_info = emoji.emoji_list(sample_emoji)
    if emoji_info[0]['match_end'] != len(sample_emoji):
        if word[-1] == sample_emoji[emoji_info[0]['match_end']]:
            sample_emoji = sample_emoji[:emoji_info[0]['match_start']] + emoji_info[0]['emoji']\
                           + sample_emoji[emoji_info[0]['match_end']:][1:]
        else:
            pass

    emoji_info = emoji.emoji_list(sample_emoji)
    sentence = sample_emoji[:emoji_info[0]['match_start']] + word + sample_emoji[emoji_info[0]['match_end']:]
    return sentence
