# emoji_text_translator🌍🐻💕👊🎁...
> emoji文本1：人生要往前看、生命一片好光景、💪一切都会越来越好的  
> ETT翻译1：&nbsp;&nbsp;&nbsp;人生要往前看、生命一片好光景、努力一切都会越来越好的  
> 
> emoji文本2：退伍证可以免费坐🚌和🚇我一直在用  
> ETT翻译2：&nbsp;&nbsp;&nbsp;退伍证可以免费坐公交车和地铁我一直在用  
> 
> emoji文本3：美国侵略📏拉克炸中国大使馆时你干啥去了  
> ETT翻译3：&nbsp;&nbsp;&nbsp;美国侵略伊拉克炸中国大使馆时你干啥去了  
### 1. 简介
本项目提供了一套将一般的带有emoji表情的中文句子完整地翻译为纯中文句子的解决方案。  
项目提出的emoji_text_translator模型（下称ETT）在[2022字节跳动安全AI挑战赛](https://security.bytedance.com/fe/2022/ai-challenge#/challenge)复赛赛道一：Emoji复杂文本识别中获得了**第六名**的成绩。ETT主要运用了ngram技术、深度学习技术（BERT预训练语言模型）等技术和常见固定搭配策略、中文词性搭配策略等特定策略来实现对emoji文本（特别是抖音黑灰产语境下的emoji文本）进行准确翻译。  
![image](https://github.com/2654400439/emoji_text_translator/blob/main/utils/1665393944382.png)
<br/>
本项目提供了一个用来处理单个一般emoji文本的脚本文件[main_handle_a_sentence.py](https://github.com/2654400439/emoji_text_translator/blob/main/main_handle_a_sentence.py)和一个专门用来处理比赛测试集的脚本文件[main_test_dataset.py](https://github.com/2654400439/emoji_text_translator/blob/main/main_test_dataset.py)，您可以根据需要自行选择。  
<br/>
**To 竞赛工作人员**: 请使用针对测试集的脚本文件进行结果验证，若使用处理单个emoji文本的脚本文件对测试集进行处理，ETT将会退化成84分左右的模型，因为竞赛过程的ETT增加了面向数据集的拼音处理模块和不同语境处理模块，该模块在一般场景下泛用性差，故在面向单个一般emoji文本的脚本文件中已经去除。面向拼音处理模块和不同语境处理模块详见[额外功能模块介绍（面向竞赛）](#competition)。
<br/>
<br/>
<br/>
> 如果您不想了解技术细节，[点击此处](#using)跳转到使用方法
### 2. ETT工作流程  
(1) **数据加载**：读取必备的数据和模型。  
(2) **句子切分**：ETT会首先判断当前句子中emoji的个数，如果emoji个数为一个则直接进入下面的标准化处理流程，如果emoji个数大于一则分别按照每一个emoji以及其对应上下文（上一个emoji（或开头）到下一个meoji（或结尾）中间的所有内容）进行切分，原句子中有多少emoji就切分成多少个子句，之后再将这些子句通过标准化处理流程进行处理，最后按先后顺序拼接。  
(3) **固定搭配检测**：ETT根据训练集数据，生成每个emoji以及上下文可能存在的固定搭配情况，之后根据阈值选择是否采信固定搭配模块输出结果，相比于下文的3gram模块，该模块可以理解为一个更加灵活的类2gram的检测模块。  
(4) **bert掩码填词**：使用bert预训练语言模型，直接将emoji句子中的emoji使用MASK遮盖处理，之后根据预测结果的最高置信分数决定是否采信该模块输出结果。  
(5) **bert单纯语言模型**：使用bert预训练语言模型，计算候选句子困惑度。使用此方法加中文词性搭配规则，实现在多个候选答案中确定出最佳的翻译结果或在必要的emoji翻译前增加对应的动词等功能。  
(6) **ngram语言模型**：根据训练集数据，对符合特定条件（出现次数较高且候选答案不集中）的每个emoji生成独自的3gram语言模型。之后使用过程中对于3gram语言模型给出候选答案超过某个阈值的则选择采信，否则直接使用候选答案中出现最频繁的翻译进行替换。  
(7) **未知emoji处理1**：使用[emoji中文网](https://www.emojiall.com/zh-hans)上提供的更全的单emoji翻译，再经过去性别化、关键部分提取等策略，来处理测试集中的emoji在训练集中未出现过的问题。  
(8) **未知emoji处理2**：对于emoji中文网上也没有出现过的emoji，直接使用python的emoji库提供的标准emoji英文翻译，再经过去性别化、去肤色化等策略，之后使用谷歌翻译api将英文翻译成中文进行emoji的翻译。  
(9) **拼音规则（面向竞赛）**：通过仔细的对训练集和测试集的观察可以发现，测试集样本是直接从训练集中随机抽取的，但是抽取结束后训练集和测试集都没有进行打乱，且最重要的是测试集样本是按照文本第一个字的拼音首字母进行排序的，这就导致对于emoji出现在开头的某些测试集样本我们可以通过获取上下文样本的拼音首字母来缩小候选答案（训练集的ground truth）的范围。  
(10) **语境选择（面向竞赛）**：通过观察测试集可以发现，id在9w之前和之后的样本明显属于两种不同的语境，故对测试集处理时我们对9w之前的样本采用标准处理，对于9w之后的样本增加了一般语境的适应。
<br/>
### 3. 功能模块介绍  
padding
<br/>
<a id="competition"></a>
### 4. 额外功能模块介绍（面向竞赛）  
padding
<br/>
<a id="using"></a> 
### 5. 运行  
运行环境：python3.9，并执行
```python
pip install -r ./requirements.txt
```
**下载BERT预训练语言模型**（不使用BERT模块则不需要），具体[参见这里](https://github.com/2654400439/emoji_text_translator/blob/main/dataset/Readme.md)。  
<br/>
<br/>
**To 竞赛工作人员**: 请使用针对测试集的脚本文件进行结果验证，若使用处理单个emoji文本的脚本文件对测试集进行处理，ETT将会退化成84分左右的模型，因为竞赛过程的ETT增加了面向数据集的拼音处理模块和不同语境处理模块，该模块在一般场景下泛用性差，故在面向单个一般emoji文本的脚本文件中已经去除。面向拼音处理模块和不同语境处理模块详见[额外功能模块介绍（面向竞赛）](#competition)。  
<br/>
#### 使用场景：（1）脚本模式（2）命令行模式（3）竞赛测试集模式（4）网站api模式
> **（1）脚本模式**
> 使用[main_handle_a_sentence.py](https://github.com/2654400439/emoji_text_translator/blob/main/main_handle_a_sentence.py)，修改函数参数以使用。  
> :param sample_emoji:  需要翻译的带有emoji的单条句子  
> :param context: 当前语境，适用于抖音黑灰产语境和一般语境[douyin(default)|normal]  
> :param fixed: 是否使用固定搭配模块[bool(dafault:True)]  
> :param bert: 是否使用bert模型增强处理能力[bool(dafault:True)]  
> <br/>
> **（2）命令行模式**
> 使用[main_handle_a_sentence.py](https://github.com/2654400439/emoji_text_translator/blob/main/main_handle_a_sentence.py)，按照文件内提示注释python程序模式，并打开命令行模式注释  
> ```python
> python main_handle_a_sentence.py --sentence SENTENCE [--context CONTEXT] [--fixed FIXED(bool)] [--bert BERT(bool)]
> ```
> 例如不使用固定搭配和bert模块，对一般语境下的句子‘你看这个像不像🐎’进行翻译：
> ```python
> python main_handle_a_sentence.py --sentence 你看这个像不像🐎 --context normal --fixed False --bert False
> ```
> **（3）竞赛测试集模式**
> 使用[main_test_dataset.py](https://github.com/2654400439/emoji_text_translator/blob/main/main_test_dataset.py)，直接运行即可，结果将保存在./result/中  
> **（4）网页api模式**
> 待实现
