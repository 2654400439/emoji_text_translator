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
(1) ngram语言模型模块  
> ETT中最基础的emoji翻译是通过ngram语言模型实现的。ngram模型能够很好地处理测试样本和训练数据集处于相同语境下的emoji翻译问题。ngram中的n选取越大则模型精度越高，但是相应的计算复杂度也会随之上升。综合考虑精度和耗时我们最终确定使用3gram。训练过程为对所有样本中除emoji翻译部分外其他文本进行分词，然后即可训练对应的3gram语言模型。测试时通过构造当前句中emoji与上文和下文的共3对3gram样本，使用训练好的3gram模型对3对样本分别计算置信度后结果取平均，如果达到阈值则采信，否则直接采用该emoji出现最频繁的翻译进行替换。ETT并非对整个训练集直接训练3gram语言模型，而是对符合特定规则的emoji分别生成独立的3gram语言模型，这正是利用ngram模型计算开销较小的优势实现了更细粒度的模型构建，而这种做法是深度学习甚至一般机器学习所难以实现的。对于选取emoji的特定规则，我们考虑需要符合两点条件：1.该emoji必须在训练集中出现超过一定次数，这样才有统计意义，否则偶然误差太大；2.该emoji在训练集中的候选翻译需要相对不集中，对于候选翻译相对集中的情况（比如该emoji出现了100次，其中99次都是对应的同一种翻译）完全可以直接采用出现最频繁的翻译直接替换，因为对于ngram模型来说如果候选翻译相对集中，则模型也会更倾向于出现次数多的翻译结果，故采用直接替换的方式，可以降低一部分时间开销。  

(2) 未知emoji处理模块  
> ETT还实现了更具泛用性的未知emoji处理模块，该模块首先解决了竞赛过程中测试集样本emoji在训练集中不曾出现过的问题，其次该模块使用基于emoji官方库的翻译处理方式完全可以在实际应用中解决处理任何可显示emoji翻译的问题。对于训练集中未出现过的emoji，ETT首先构建了基于emoji中文网单emoji翻译的增强版数据仓库（包含常见的4.5k个emoji），通过对该类数据进行关键信息抽取，我们提取到了可用的emoji中文翻译。若测试集样本中的emoji在emoji中文网中仍未出现过，则使用python的emoji官方库，能将任何可显示的emoji翻译成标准的英文译文，之后通过去除肤色、去除性别干扰等处理步骤后采用英译汉处理，就获得了更一般的通用emoji中文翻译。这种方法虽然翻译完善程度较低，但是能解决任何emoji的翻译问题。  

(3) 固定搭配检测模块
> 随后我们又通过观测训练集，发现有较多的表情是和其前一个Token或后一个Token形成较为准确固定搭配的，因此ETT又离线统计了训练集中一个表情所出现的所有二元固定搭配（A + Emoji 或 Emoji + A的形式，其中 A 为中文或数字或字母）以及这个固定搭配中表情对应的翻译分布，共1327个表情得到了统计。随后针对测试集样本，针对样本中每个表情，提取该表情的前向固定搭配（A+Emoji)与后向固定搭配(Emoji+A)，并通过查表的形式得到这个表情对应所有可能翻译以及频数，选择频率最打的翻译作为最终的结果。但是为了应对训练数据集中噪音的情况，我们对最终结果的翻译频率作了限制，要求其频率必须高于 0.66 如果只有一个候选翻译结果则要求其频数高于 1，否则过滤该样本，不作处理。后来又发现前向和后向固定搭配并不是同等优先级，有的表情前向固定搭配单一，频数较大，而后向固定搭配混杂，频数平均，这样如果综合平等考虑前向后向固定搭配会导致这些样本无法识别处理因此对于综合考虑前向后向固定搭配无法处理的样本，我们又单独分开考虑前向和后向固定搭配的结果，按照频率最高的答案作为候选，并过滤掉最高频率低于 0.66 以及单一频数低于 2 的样本

(4) bert预训练语言模型模块  
> ETT通过深度学习模型进一步提高翻译能力。ETT并没有直接采用深度学习模型来构建一个端到端的emoji翻译解决方案或者是第四范式prompt模型等，这是因为我们前期进行了如是的尝试，但是效果不理想，甚至不如全部直接使用出现最频繁的翻译进行替换得到的分数高。分析其原因我们认为首先是测试集的数据分布和训练集并不完全相同，测试集id前9w条和9w条之后的样本语境完全不同，如果使用训练集数据来训练一个端到端的模型则会面临概念漂移的问题；其次是接近一般的训练集样本长度较短，这导致深度学习模型能捕获的上下文语境信息不足，进一步导致了处理翻译时注意力分散的问题。综上，我们最终使用ngram模型作为ETT的基础翻译处理模型，bert模型用来辅助处理一些语义、词性或知识库固定搭配的问题。具体来说，bert模型在ETT中的第一个用处就是基于预训练知识的掩码填词来处理一些人类知识库固定搭配的问题，能处理诸如‘📏拉克’翻译成‘伊拉克’的问题（该emoji在训练集中候选翻译中并不包含伊），当掩码填词模块返回置信度高于阈值时我们予以采信。其次，bert模型还被用于解决emoji动名词翻译的问题（‘我今天🚲来上班的’需要将emoji翻译成‘骑’+‘自行车’；‘等下我们去🛒’需要将emoji翻译成‘购物车’-‘车’），该模块通过增减动词后使用预训练语言模型来计算句子的困惑度，通过困惑度是否降低来判断句子是否更通顺，以此来确定是否采信该模块翻译。进一步的，bert模型还被用来做限制条件下的候选翻译选择。如对于中文形容词搭配的处理，如果emoji前一个字是‘很’，则考虑该emoji应该翻译为一个形容词，之后通过在候选词典里筛选形容词，再使用预训练语言模型的句子困惑度来选择合适的翻译。  
<br/>

### 4. 额外功能模块介绍（面向竞赛）  
<a id="competition"></a> 
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
