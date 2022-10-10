from ngram_model.ngram import generate_handle_list, generate_lm_list
from pinyin_model.pinyin_utils import generate_pinyin_initial_list
import pickle


def dump_lm_list():
    handle_list = generate_handle_list()
    lm_list = generate_lm_list(handle_list)

    with open('../dataset/lm_list.data', 'wb') as f:
        pickle.dump(lm_list, f)


def load_lm_list(file_path):
    with open(file_path, 'rb') as f:
        lm_list = pickle.load(f)
    return lm_list


def dump_pinyin_list():
    initial_list = generate_pinyin_initial_list()
    with open('../dataset/pinyin_list.data', 'wb') as f:
        pickle.dump(initial_list, f)


def load_pinyin_list(file_path):
    with open(file_path, 'rb') as f:
        pinyin_list = pickle.load(f)
    return pinyin_list


def load_pickle_data_from_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


if __name__ == '__main__':
    dump_pinyin_list()
