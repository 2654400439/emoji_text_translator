from nltk.translate.bleu_score import sentence_bleu
import Levenshtein
from math import pow


def get_bleu(sentence1, sentence2):
    tmp1 = [sentence1[i] for i in range(len(sentence1))]
    tmp2 = [sentence2[i] for i in range(len(sentence2))]
    return sentence_bleu([tmp1], tmp2, weights=(0, 1, 0, 0))


def get_WER(sentence1, sentence2):
    """
    sentence1是提交答案，sentence2是参考答案
    """
    return 1 - Levenshtein.distance(sentence1, sentence2)/len(sentence2)


def get_score(sentence1, sentence2):
    bleu = get_bleu(sentence1, sentence2)
    WER = get_WER(sentence1, sentence2)
    return pow(50, bleu) + pow(50, WER)


def demo():
    # 测试样例

    # 对于超短样本测试
    # 前方🛑  --> reference：前方红灯
    print('--------------------------------对于超短样本----------------------------------')
    print('情况一：完全预测正确，分数：', get_score('前方红灯', '前方红灯'))
    print('情况二：完全不做处理，分数：', get_score('前方🛑', '前方红灯'))
    print('情况三：直接去除emoji，分数：', get_score('前方', '前方红灯'))
    print('情况四（1）：按照越短越好的策略预测，且对了一部分，分数：', get_score('前方红', '前方红灯'))
    print('情况四（2）：按照越短越好的策略预测，且对了一部分，分数：', get_score('前方灯', '前方红灯'))
    print('情况四（3）：按照越短越好的策略预测，但没预测对，分数：', get_score('前方虎', '前方红灯'))
    print('情况五：预测字数正确，但驴头不对马嘴，分数：', get_score('前方老虎', '前方红灯'))
    print('情况六：预测字数较多，且完全不对，分数：', get_score('前方有只老虎', '前方红灯'))
    print('情况七：预测字数较多，但包含正确答案，分数：', get_score('前方红灯有只', '前方红灯'))
    # 对比情况四和情况七，即使情况七预测到了正确答案，但是字数超了依然得了不高的分数，且万一预测中不包含正确答案，分数会更低
    # 结论：建议在预测把握不大的情况下，考虑使用越短越好的策略
    print('\n\n')
    # 对于标准样本测试
    # 军人成为某些政治🤡的牺牲品，悲壮！  --> reference：军人成为某些政治小丑的牺牲品，悲壮！
    print('--------------------------------对于标准样本----------------------------------')
    print('情况一：完全预测正确，分数：', get_score('军人成为某些政治小丑的牺牲品，悲壮！', '军人成为某些政治小丑的牺牲品，悲壮！'))
    print('情况二：完全不做处理，分数：', get_score('军人成为某些政治🤡的牺牲品，悲壮！', '军人成为某些政治小丑的牺牲品，悲壮！'))
    print('情况三：直接去除emoji，分数：', get_score('军人成为某些政治的牺牲品，悲壮！', '军人成为某些政治小丑的牺牲品，悲壮！'))
    print('情况四（1）：按照越短越好的策略预测，且对了一部分，分数：', get_score('军人成为某些政治小的牺牲品，悲壮！', '军人成为某些政治小丑的牺牲品，悲壮！'))
    print('情况四（2）：按照越短越好的策略预测，且对了一部分，分数：', get_score('军人成为某些政治丑的牺牲品，悲壮！', '军人成为某些政治小丑的牺牲品，悲壮！'))
    print('情况四（3）：按照越短越好的策略预测，但没预测对，分数：', get_score('军人成为某些政治大的牺牲品，悲壮！', '军人成为某些政治小丑的牺牲品，悲壮！'))
    print('情况五：预测字数正确，但驴头不对马嘴，分数：', get_score('军人成为某些政治老虎的牺牲品，悲壮！', '军人成为某些政治小丑的牺牲品，悲壮！'))
    print('情况六：预测字数较多，且完全不对，分数：', get_score('军人成为某些政治大老虎的牺牲品，悲壮！', '军人成为某些政治小丑的牺牲品，悲壮！'))
    print('情况七：预测字数较多，但包含正确答案，分数：', get_score('军人成为某些政治大小丑的牺牲品，悲壮！', '军人成为某些政治小丑的牺牲品，悲壮！'))


if __name__ == '__main__':
    demo()