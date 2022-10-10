from preprocess.load_data import generate_emoji_dict
from translate import Translator
import emoji


class LanguageTrans():
    def __init__(self, mode):
        self.mode = mode
        if self.mode == "E2C":
            self.translator = Translator(from_lang="english", to_lang="chinese")
        if self.mode == "C2E":
            self.translator = Translator(from_lang="chinese", to_lang="english")

    def trans(self, word):
        translation = self.translator.translate(word)
        return translation


def emoji2english2chinese(sample_emoji):
    translator = LanguageTrans("E2C")

    emoji_dict = generate_emoji_dict()

    emoji_english = emoji.demojize(emoji.emoji_list(sample_emoji)[0]['emoji'])[1:-1]
    # 处理带有皮肤信息的样本
    if len(emoji_english) > 12:
        if emoji_english[-9:] == 'skin_tone':
            if emoji.emojize(':'+'_'.join(emoji_english.split('_')[:-3])+':') in emoji_dict:
                word = emoji_dict[emoji.emojize(':'+'_'.join(emoji_english.split('_')[:-3])+':')]
            else:
                word = translator.trans('_'.join(emoji_english.split('_')[:-3]).split('_')[-1])
        else:
            word = translator.trans(emoji_english.split('_')[-1])
    else:
        word = translator.trans(emoji_english.split('_')[-1])
    if len(word) > 4:
        word = word[4:]

    return word


if __name__ == '__main__':
    print(emoji2english2chinese('有个🎁送给你'))