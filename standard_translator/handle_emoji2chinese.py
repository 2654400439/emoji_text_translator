import emoji
from jieba import posseg


def emoji2chinese_dict(sample_emoji, data_chinese_dict, emoji_dict):
    """
    对于训练集中没有出现过的emoji，使用emojiall数据进行翻译替换
    :param sample_emoji:  测试集样本
    :param data_chinese_dict:  字典类型的emojiall数据
    :param emoji_dict:
    :return: emojiall单emoji翻译
    """
    emoji_english = emoji.demojize(emoji.emoji_list(sample_emoji)[0]['emoji'])[1:-1]
    emoji_current = emoji.emoji_list(sample_emoji)[0]['emoji']
    chinese = data_chinese_dict[emoji_current]

    # 处理带有skin_tone的样本
    if len(emoji_english) > 12:
        if emoji_english[-9:] == 'skin_tone':
            if emoji.emojize(':' +'_'.join(emoji_english.split('_')[:-3] ) +':') in emoji_dict:
                word = emoji_dict[emoji.emojize(':' +'_'.join(emoji_english.split('_')[:-3] ) +':')]
                return word

    # 去除性别干扰翻译
    if chinese != '男人' and chinese != '男孩' and chinese != '女人' and chinese != '女孩':
        chinese = chinese.replace('的男人' ,'').replace('男子' ,'').replace('男人' ,'').replace('男' ,'').replace('的人' ,'')
        chinese = chinese.replace('的女人' ,'').replace('女子' ,'').replace('女人' ,'').replace('女' ,'')

    # 根据词性，名词前后不能再是名词
    # 新加的0926判断长度大于3
    if len(sample_emoji) > 3:
        if len(list(posseg.cut(chinese))) == 1 and list(posseg.cut(chinese))[0].flag == 'n':
            # emoji不在开头就对左侧进行词性判断
            if emoji.emoji_list(sample_emoji)[0]['match_start'] != 0:
                left = sample_emoji[:emoji.emoji_list(sample_emoji)[0]['match_start']]
                if list(posseg.cut(left))[-1].flag == 'n':
                    word = ''
                    return word
            # emoji不在结尾就对右侧进行词性判断
            if emoji.emoji_list(sample_emoji)[0]['match_end'] != len(sample_emoji):
                right = sample_emoji[emoji.emoji_list(sample_emoji)[0]['match_end']:]
                if len(list(posseg.cut(chinese+right))) != 1:
                    if list(posseg.cut(right))[0].flag == 'n':
                        word = ''
                        return word

    # 对于还存在修饰性词汇的，直接删除修饰词
    if chinese.find('的') != -1:
        chinese = chinese.split('的')[-1]

    # 对于长度还超过4的直接舍弃
    if len(chinese) > 4:
        chinese = ''

    return chinese
