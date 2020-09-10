import json
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer


def load_data(file_name, tokenizer, label_seq):
    steps = 0
    data = []
    nums = np.zeros([len(label_seq)])
    c_num = 0
    max_len = 0
    with open(file_name, 'r', encoding='utf8')as fp:
        json_data = json.load(fp)
        for episode in json_data["episodes"]:
            for scenes in episode["scenes"]:
                tokens = [101]
                emotions = []
                sep_place = [0]
                for uttr in scenes["utterances"]:
                    us = tokenizer.encode(uttr["transcript"])[1:-1]
                    if len(tokens) + len(us) + 1 < 512:
                        tokens += us
                        tokens.append(102)
                        k = uttr["emotion"]
                        if k in ["Neutral", "Joyful", "Sad", "Mad"]:
                            emotions.append(k)
                            nums[label_seq.index(k)] += 1
                        else:
                            emotions.append("Out-Of-Domain")
                            nums[label_seq.index("Out-Of-Domain")] += 1
                        sep_place.append(len(tokens))
                        c_num += 1
                    else:
                        break
                if len(tokens) > max_len:
                    max_len = len(tokens)
                m = []
                for i, emotion in enumerate(emotions):
                    vec = np.zeros([len(label_seq)])
                    vec[label_seq.index(emotion)] = 1.
                    m.append([tf.convert_to_tensor(vec, dtype=np.float32), sep_place[i:i + 2]])
                    steps += 1
                data.append([tokens, m])

    print("Data num: {}".format(len(data)))
    print("Max len: {}".format(max_len))
    frequency = nums/np.sum(nums)
    return data, frequency, steps, nums


def new_load_data(file_name, tokenizer, label_seq):
    steps = 0
    data = []
    nums = np.zeros([len(label_seq)])
    c_num = 0
    max_len = 0
    with open(file_name, 'r', encoding='utf8')as fp:
        json_data = json.load(fp)
        for scene in json_data:
            tokens = [101]
            emotions = []
            sep_place = [0]
            for uttr in scene:
                us = tokenizer.encode(uttr["utterance"])[1:-1]
                if len(tokens) + len(us) + 1 < 512:
                    tokens += us
                    tokens.append(102)
                    k = uttr["emotion"]
                    if k in ["neutral", "joy", "sadness", "anger"]:
                        emotions.append(k)
                        nums[label_seq.index(k)] += 1
                    else:
                        emotions.append("out-of-domain")
                        nums[label_seq.index("out-of-domain")] += 1
                    sep_place.append(len(tokens))
                    c_num += 1
                else:
                    break
            if len(tokens) > max_len:
                max_len = len(tokens)
            m = []
            for i, emotion in enumerate(emotions):
                vec = np.zeros([len(label_seq)])
                vec[label_seq.index(emotion)] = 1.
                m.append([tf.convert_to_tensor(vec, dtype=np.float32), sep_place[i:i + 2]])
                steps += 1
            data.append([tokens, m])

        print("Data num: {}".format(len(data)))
        print("Max len: {}".format(max_len))
        print(c_num)
        frequency = nums / np.sum(nums)
        return data, frequency, steps, nums


def load_augmented_data(file_name, tokenizer, label_seq):
    steps = 0
    data = []
    nums = np.zeros([len(label_seq)])
    c_num = 0
    max_len = 0
    with open(file_name, "r") as f:
        dialogues = json.load(f)
        for dialogue in dialogues:
            tokens = [101]
            emotions = []
            sep_place = [0]
            tokens_de = [101]
            sep_place_de = [0]
            tokens_fr = [101]
            sep_place_fr = [0]
            tokens_it = [101]
            sep_place_it = [0]
            for uttr in dialogue:
                if len(uttr["utterance"]) == 0:
                    continue
                us = tokenizer.encode(uttr["utterance"])[1:-1]
                us_de = tokenizer.encode(uttr["utterance_de"])[1:-1]
                us_fr = tokenizer.encode(uttr["utterance_fr"])[1:-1]
                us_it = tokenizer.encode(uttr["utterance_it"])[1:-1]
                if len(tokens) + len(us) + 1 < 512:
                    tokens += us
                    tokens.append(102)
                    k = uttr["emotion"]
                    if k in ["neutral", "joy", "sadness", "anger"]:
                        emotions.append(k)
                        nums[label_seq.index(k)] += 1
                    else:
                        emotions.append("out-of-domain")
                        nums[label_seq.index("out-of-domain")] += 1
                    sep_place.append(len(tokens))
                    c_num += 1
                else:
                    break

                if len(tokens_de) + len(us_de) + 1 < 512:
                    tokens_de += us_de
                    tokens_de.append(102)
                    sep_place_de.append(len(tokens_de))
                    c_num += 1
                else:
                    break

                if len(tokens_fr) + len(us_fr) + 1 < 512:
                    tokens_fr += us_fr
                    tokens_fr.append(102)
                    sep_place_fr.append(len(tokens_fr))
                    c_num += 1
                else:
                    break

                if len(tokens_it) + len(us_it) + 1 < 512:
                    tokens_it += us_it
                    tokens_it.append(102)
                    sep_place_it.append(len(tokens_it))
                    c_num += 1
                else:
                    break
            if len(tokens) > max_len:
                max_len = len(tokens)
            if len(tokens_de) > max_len:
                max_len = len(tokens_de)
            if len(tokens_fr) > max_len:
                max_len = len(tokens_fr)
            if len(tokens_it) > max_len:
                max_len = len(tokens_it)
            ms = []
            ms_de = []
            ms_fr = []
            ms_it = []
            for i, emotion in enumerate(emotions):
                vec = np.zeros([len(label_seq)])
                vec[label_seq.index(emotion)] = 1.
                k = tf.convert_to_tensor(vec, dtype=np.float32)
                ms.append([k, sep_place[i:i + 2]])
                ms_de.append([k, sep_place_de[i:i + 2]])
                ms_fr.append([k, sep_place_fr[i:i + 2]])
                ms_it.append([k, sep_place_it[i:i + 2]])
                steps += 4
            data.append([tokens, ms])
            data.append([tokens_de, ms_de])
            data.append([tokens_fr, ms_fr])
            data.append([tokens_it, ms_it])
    print("Data num: {}".format(len(data)))
    print("Max len: {}".format(max_len))
    frequency = nums / np.sum(nums)
    return data, frequency, steps, nums


def load_label_seq(file_name):
    with open(file_name, "r", encoding="utf8") as f:
        k = f.readline()
        k = k.split("\t")
        return k


def load_label_data(context_file, emotion_file, tokenizer, label_seq):
    cf = open(context_file, "r")
    ef = open(emotion_file, "r")
    nums = np.zeros([len(label_seq)])
    c_num = 0
    max_len = 0
    steps = 0
    data = []
    for s_line, e_line in zip(cf, ef):
        sentences = s_line.strip().split("\t")
        emotions = e_line.strip().split("\t")
        assert len(sentences) == len(emotions)
        tokens = [101]
        f_emotions = []
        sep_place = [0]
        for i, sentence in enumerate(sentences):
            us = tokenizer.encode(sentence)[1: -1]
            if len(tokens) + len(us) + 1 < 512:
                tokens += us
                tokens.append(102)
                k = emotions[i]
                if k in ["n", "j", "sa", "su"]:
                    f_emotions.append(k)
                    nums[label_seq.index(k)] += 1
                else:
                    f_emotions.append("out-of-domain")
                    nums[label_seq.index("out-of-domain")] += 1
                sep_place.append(len(tokens))
                c_num += 1
            else:
                break
        if len(tokens) > max_len:
            max_len = len(tokens)
        m = []
        for i, emotion in enumerate(f_emotions):
            vec = np.zeros([len(label_seq)])
            vec[label_seq.index(emotion)] = 1.
            m.append([tf.convert_to_tensor(vec, dtype=np.float32), sep_place[i:i + 2]])
            steps += 1
        data.append([tokens, m])
    print("Data num: {}".format(len(data)))
    print("Max len: {}".format(max_len))
    print(c_num)
    frequency = nums / np.sum(nums)
    return data, frequency, steps, nums


def persona_label_data(context_file, emotion_file, your_persona_file, partner_persona_file, tokenizer, label_seq):
    cf = open(context_file, "r")
    ef = open(emotion_file, "r")
    ypf = open(your_persona_file, "r")
    ppf = open(partner_persona_file, "r")
    nums = np.zeros([len(label_seq)])
    c_num = 0
    max_len = 0
    steps = 0
    data = []
    for s_line, e_line, yp_line, pp_line in zip(cf, ef, ypf, ppf):
        sentences = s_line.strip().split("\t")
        emotions = e_line.strip().split("\t")
        yps = yp_line.strip().split("\t")
        pps = pp_line.strip().split("\t")
        assert len(sentences) == len(emotions)
        tokens = [101]
        yp_tokens = [101]
        pp_tokens = [101]
        f_emotions = []
        sep_place = [0]
        yp_sep_place = [0]
        pp_sep_place = [0]
        for sentence in yps:
            us = tokenizer.encode(sentence[16:])[1: -1]
            if len(yp_tokens) + len(us) + 1 < 512:
                yp_tokens += us
                yp_tokens.append(102)
                yp_sep_place.append(len(yp_tokens))
        for sentence in pps:
            us = tokenizer.encode(sentence[21:])[1: -1]
            if len(pp_tokens) + len(us) + 1 < 512:
                pp_tokens += us
                pp_tokens.append(102)
                pp_sep_place.append(len(pp_tokens))
        for i, sentence in enumerate(sentences):
            us = tokenizer.encode(sentence)[1: -1]
            if len(tokens) + len(us) + 1 < 512:
                tokens += us
                tokens.append(102)
                k = emotions[i]
                if k in ["n", "j", "sa", "su"]:
                    f_emotions.append(k)
                    nums[label_seq.index(k)] += 1
                else:
                    f_emotions.append("out-of-domain")
                    nums[label_seq.index("out-of-domain")] += 1
                sep_place.append(len(tokens))
                c_num += 1
            else:
                break
        if len(tokens) > max_len:
            max_len = len(tokens)
        m = []
        for i, emotion in enumerate(f_emotions):
            vec = np.zeros([len(label_seq)])
            vec[label_seq.index(emotion)] = 1.
            m.append([tf.convert_to_tensor(vec, dtype=np.float32), sep_place[i:i + 2]])
            steps += 1
        data.append([tokens, m, yp_tokens, yp_sep_place, pp_tokens, pp_sep_place])
    print("Data num: {}".format(len(data)))
    print("Max len: {}".format(max_len))
    print(c_num)
    frequency = nums / np.sum(nums)
    return data, frequency, steps, nums


if __name__ == "__main__":
    '''label_seq = load_label_seq("label_seq.txt")
    token = BertTokenizer.from_pretrained("bert-base-uncased")
    d, f, s = load_data("./data/emotion-detection-trn.json", token, label_seq)
    print(d[0])
    #print(label2vec(l, label_seq))
    token = BertTokenizer.from_pretrained("bert-base-uncased")
    label_seq = load_label_seq("new_label_seq.txt")
    d, f, _ = new_load_data("Friends/friends_dev.json", token, label_seq)
    print(f)'''
    '''token = BertTokenizer.from_pretrained("bert-base-uncased")
    label_seq = load_label_seq("new_label_seq.txt")
    d, f, _, nums = load_augmented_data("emotionpush.augmented.json", token, label_seq)
    print(f)
    token = BertTokenizer.from_pretrained("bert-base-uncased")
    label_seq = load_label_seq("new_label_seq.txt")
    d, f, _, nums = load_augmented_data("emotionpush.augmented.json", token, label_seq)
    print(f)
    token = BertTokenizer.from_pretrained("bert-base-uncased")
    label_seq = load_label_seq("new_label_seq.txt")
    d, f, _, _ = new_load_data("Friends/friends.augmented.json", token, label_seq)
    token = BertTokenizer.from_pretrained("bert-base-uncased")
    label_seq = load_label_seq("dialogue_label_seq.txt")
    d, f, _, _ = persona_label_data("./dialogue_data/train_contexts.txt", "./dialogue_data/train_emotions.txt", "./dialogue_data/train_your_personas.txt", "./dialogue_data/train_partner_personas.txt", token, label_seq)'''
    with  open("./dialogue_data/train_your_personas.txt") as f:
        k = 0
        for line in f:
            k+=1
        print(k)