import emoji
from preprocess.load_data import generate_emoji2chinese_dict


def handle_not_emoji(sample_emoji):
    emojiall_data = generate_emoji2chinese_dict()
    emojiall_keys = list(emojiall_data.keys())
    if len(emoji.emoji_list(sample_emoji)) == 0:
        current_emoji = sample_emoji[-2:]
        for i in range(len(emojiall_keys)):
            if current_emoji in emojiall_keys[i]:
                chinese = emojiall_data[emojiall_keys[i]]
                chinese = chinese.replace('男生', '').replace('女生', '').replace('男', '').replace('女', '')
                return sample_emoji[:-2] + chinese
    return sample_emoji





