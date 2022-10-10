# emoji_text_translator
### 简介
本项目提供了一套将一般的带有emoji表情的中文句子完整地翻译为纯中文句子的解决方案。  
项目提出的emoji_text_translator模型（下称ETT）在[2022字节跳动安全AI挑战赛](https://security.bytedance.com/fe/2022/ai-challenge#/challenge)复赛赛道一：Emoji复杂文本识别中获得了**第六名**的成绩。ETT主要运用了ngram技术、深度学习技术（BERT预训练语言模型）等技术和常见固定搭配策略、中文词性搭配策略等特定策略来实现对emoji文本（特别是抖音黑灰产语境下的emoji文本）进行准确翻译。  
<br/>
本项目提供了一个用来处理单个一般emoji文本的脚本文件[main_handle_a_sentence.py](https://github.com/2654400439/emoji_text_translator/blob/main/main_handle_a_sentence.py)和一个专门用来处理比赛测试集的脚本文件[main_test_dataset.py](https://github.com/2654400439/emoji_text_translator/blob/main/main_test_dataset.py)，您可以根据需要自行选择。  
**To 竞赛工作人员：**请使用针对测试集的脚本文件进行结果验证，若使用处理单个emoji文本的脚本文件对测试集进行处理，ETT将会退化成84分左右的模型
