from transformers import BertTokenizer, BertForMaskedLM


def initialize_bert():
    model = BertForMaskedLM.from_pretrained("./dataset/bert_base_chinese")
    tokenizer = BertTokenizer.from_pretrained("./dataset/bert_base_chinese")
    return model, tokenizer
