import emoji


def handle_same_emoji(sample_emoji):
    """
    处理单句中有两个连续相同的emoji的情况
    :param sample_emoji: 包含emoji的测试集样本
    :return: 将连续相同emoji变成一个emoji的句子
    """
    if emoji.emoji_list(sample_emoji)[0]['emoji'] == emoji.emoji_list(sample_emoji)[1]['emoji']:
        if emoji.emoji_list(sample_emoji)[0]['match_end'] == emoji.emoji_list(sample_emoji)[1]['match_start']:
            sample_emoji = sample_emoji[:emoji.emoji_list(sample_emoji)[0]['match_start']] + \
                           sample_emoji[emoji.emoji_list(sample_emoji)[1]['match_start']:]
    return sample_emoji