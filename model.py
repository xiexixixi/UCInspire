import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformers import BertTokenizer, TFBertModel, BertConfig
from tools import *
from random import shuffle
import logging
logging.disable(30)

import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='Utility Functions')
parser.add_argument('-load_post_trained_bert', default=False)
parser.add_argument('-dataset', type=str, default="./dialogue_data/friends_train.json")
parser.add_argument('-evalset', type=str, default="./Friends/friends_dev.json")
parser.add_argument('-testset', type=str, default="./Friends/friends_test.json")
parser.add_argument('-post_trained_bert_file', type=str, default="./model/")
parser.add_argument('-save_path', type=str, default='./saved_model/')
parser.add_argument('-mode', type=str, default="train", choices=["train", "test"])
parser.add_argument('-pooling_way', type=str, default="max", choices=["max", "mean"])
parser.add_argument('-batch_size', type=int, default=1)
parser.add_argument('-dropout', type=float, default=0.1)
parser.add_argument('-epoch', type=int, default=10)
parser.add_argument('-max_lr', type=float, default=2e-5)
parser.add_argument('-max_grad_norm', type=float, default=5., help="prefix of .dict and .labels files")
parser.add_argument("--adam_epsilon", default=1e-7, type=float, help="Epsilon for Adam optimizer.")



args = parser.parse_args()


class PostTrainedBert(layers.Layer):
    def __init__(self, load_post_trained_bert, post_trained_bert_file):
        super(PostTrainedBert, self).__init__()
        if load_post_trained_bert:
            self.bert = TFBertModel.from_pretrained(post_trained_bert_file, from_pt=True)
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        else:
            self.bert = TFBertModel.from_pretrained("bert-base-uncased")
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def call(self, inputs, token_type_ids=None):
        return self.bert(inputs, token_type_ids=token_type_ids)


class PosEmotionXModel(keras.Model):
    def __init__(self, load_post_trained_bert, post_trained_bert_file, dropout_rate, pooling_way, mode):
        super(PosEmotionXModel, self).__init__()
        self.bert = PostTrainedBert(load_post_trained_bert, post_trained_bert_file)
        if pooling_way == "max":
            self.pooling = layers.GlobalMaxPooling1D()
        else:
            self.pooling = layers.GlobalAveragePooling1D()
        self.classifier_part1 = keras.models.Sequential()
        self.classifier_part1.add(keras.Input(shape=(768,)))
        self.classifier_part1.add(layers.Dense(384, activation='selu'))
        self.dropout = layers.Dropout(dropout_rate)
        self.classifier_part2 = keras.models.Sequential()
        self.classifier_part2.add(keras.Input(shape=(384,)))
        self.classifier_part2.add(layers.Dense(5, activation='softmax'))
        self.mode = mode

    def get_pos(self, inputs, sep_positions):
        pos = [0]*(sep_positions[0]+1)+[1]*(sep_positions[1]-sep_positions[0]-1)+[0]*(int(inputs.get_shape()[1])-sep_positions[1])
        pos = tf.convert_to_tensor(np.array(pos)[np.newaxis, :], dtype=np.int32)
        return pos

    def call(self, inputs, sep_positions, training=False):
        #pos = self.get_pos(inputs, sep_positions)
        embeddings = self.bert(inputs, token_type_ids=None)[0]
        embedding = embeddings[0]
        sentence_embed = []
        if self.mode == "emotion_classification":
            for sep_position in sep_positions:
                k = embedding[sep_position[0]: sep_position[1]]
                k = k[np.newaxis, :]
                k = self.pooling(k)[0]
                sentence_embed.append(k)
            sentence_embed = tf.stack(sentence_embed, axis=0)
            # sentence_embed = sentence_embed[np.newaxis, :]
            # pooled_embed = self.pooling(sentence_embed)
            classified_result = self.classifier_part1(sentence_embed)
            classified_result = self.dropout(classified_result, training=training)
            classified_result = self.classifier_part2(classified_result)
            return classified_result
        else:
            k = embedding[sep_positions[-1][0]: sep_positions[-1][1]]
            k = k[np.newaxis, :]
            k = self.pooling(k)[0]
            return k


class RNNPosEmotionXModel(keras.Model):
    def __init__(self, load_post_trained_bert, post_trained_bert_file, dropout_rate):
        super(RNNPosEmotionXModel, self).__init__()
        self.bert = PostTrainedBert(load_post_trained_bert, post_trained_bert_file)
        self.rnn = layers.LSTM(768, dropout=dropout_rate)
        self.classifier_part1 = keras.models.Sequential()
        self.classifier_part1.add(keras.Input(shape=(768,)))
        self.classifier_part1.add(layers.Dense(384, activation='selu'))
        self.dropout = layers.Dropout(dropout_rate)
        self.classifier_part2 = keras.models.Sequential()
        self.classifier_part2.add(keras.Input(shape=(384,)))
        self.classifier_part2.add(layers.Dense(5, activation='softmax'))

    def get_pos(self, inputs, sep_positions):
        pos = [0]*(sep_positions[0]+1)+[1]*(sep_positions[1]-sep_positions[0]-1)+[0]*(int(inputs.get_shape()[1])-sep_positions[1])
        pos = tf.convert_to_tensor(np.array(pos)[np.newaxis, :], dtype=np.int32)
        return pos

    def call(self, inputs, sep_positions, training=False):
        #pos = self.get_pos(inputs, sep_positions)
        embeddings = self.bert(inputs, token_type_ids=None)[0]
        #embeddings = self.bert(inputs)[0]
        embedding = embeddings[0]
        sentence_embed = []
        for sep_position in sep_positions:
            k = embedding[sep_position[0]: sep_position[1]]
            k = k[np.newaxis, :]
            rnn_embed = self.rnn(k, training=training)[0]
            sentence_embed.append(rnn_embed)
        sentence_embed = tf.stack(sentence_embed, axis=0)
        classified_result = self.classifier_part1(sentence_embed)
        classified_result = self.dropout(classified_result, training=training)
        classified_result = self.classifier_part2(classified_result)
        return classified_result


class PersonaEmotionXModel(keras.Model):
    def __init__(self, load_post_trained_bert, post_trained_bert_file, dropout_rate, pooling_way, mode):
        super(PersonaEmotionXModel, self).__init__()
        self.bert = PostTrainedBert(load_post_trained_bert, post_trained_bert_file)
        if pooling_way == "max":
            self.pooling = layers.GlobalMaxPooling1D()
        else:
            self.pooling = layers.GlobalAveragePooling1D()
        self.classifier_part1 = keras.models.Sequential()
        self.classifier_part1.add(keras.Input(shape=(1536,)))
        self.classifier_part1.add(layers.Dense(384, activation='selu'))
        self.dropout = layers.Dropout(dropout_rate)
        self.classifier_part2 = keras.models.Sequential()
        self.classifier_part2.add(keras.Input(shape=(384,)))
        self.classifier_part2.add(layers.Dense(5, activation='softmax'))
        self.mode = mode

    def attn(self, sentence_embed, persona_embed):
        weight = tf.nn.softmax(tf.matmul(sentence_embed, tf.transpose(persona_embed)))
        weight = tf.transpose(tf.stack([weight[0]]*persona_embed.get_shape()[1]))
        result = weight * persona_embed
        return tf.reduce_sum(result, axis=0)

    def call(self, inputs, sep_positions, yps, yp_sep_place, pps, pp_sep_place, training=False):
        #pos = self.get_pos(inputs, sep_positions)
        embeddings = self.bert(inputs, token_type_ids=None)[0]
        yp_embeddings = self.bert(yps, token_type_ids=None)[0]
        pp_embeddings = self.bert(pps, token_type_ids=None)[0]
        embedding = embeddings[0]
        yp_embedding = yp_embeddings[0]
        pp_embedding = pp_embeddings[0]
        sentence_embed = []
        attended_persona_embeds = []
        if self.mode == "emotion_classification":
            for i, sep_position in enumerate(sep_positions):
                k = embedding[sep_position[0]: sep_position[1]]
                k1 = k[np.newaxis, :]
                k2 = self.pooling(k1)
                k3 = k2[0]
                sentence_embed.append(k3)
                if i%2 == 0:
                    attended_embed = self.attn(k2, pp_embedding)
                else:
                    attended_embed = self.attn(k2, yp_embedding)
                attended_persona_embeds.append(attended_embed)
            sentence_embed = tf.stack(sentence_embed, axis=0)
            attended_persona_embeds = tf.stack(attended_persona_embeds, axis=0)
            final_embed = tf.concat([sentence_embed, attended_persona_embeds], axis=1)

            # sentence_embed = sentence_embed[np.newaxis, :]
            # pooled_embed = self.pooling(sentence_embed)
            classified_result = self.classifier_part1(final_embed)
            classified_result = self.dropout(classified_result, training=training)
            classified_result = self.classifier_part2(classified_result)
            return classified_result
        else:
            k = embedding[sep_positions[-1][0]: sep_positions[-1][1]]
            k = k[np.newaxis, :]
            k = self.pooling(k)[0]
            return k

'''class EmotionXModel(keras.Model):
    def __init__(self, dropout_rate, pooling_way):
        super(EmotionXModel, self).__init__()
        if pooling_way == "max":
            self.pooling = layers.GlobalMaxPooling1D()
        else:
            self.pooling = layers.GlobalAveragePooling1D()
        self.classifier_part1 = keras.models.Sequential()
        self.classifier_part1.add(keras.Input(shape=(768,)))
        self.classifier_part1.add(layers.Dense(384, activation='selu'))
        self.dropout = layers.Dropout(dropout_rate)
        self.classifier_part2 = keras.models.Sequential()
        self.classifier_part2.add(keras.Input(shape=(384,)))
        self.classifier_part2.add(layers.Dense(5, activation='softmax'))

    def call(self, inputs, sep_positions, training=False):
        embeddings = inputs[0]
        #embeddings = self.bert(inputs)[0]
        embedding = embeddings[0]
        sentence_embed = embedding[sep_positions[0]+1: sep_positions[1]]
        sentence_embed = sentence_embed[np.newaxis, :]
        pooled_embed = self.pooling(sentence_embed)
        classified_result = self.classifier_part1(pooled_embed)
        classified_result = self.dropout(classified_result, training=training)
        classified_result = self.classifier_part2(classified_result)
        return classified_result


class RNNEmotionXModel(keras.Model):
    def __init__(self, dropout_rate):
        super(RNNEmotionXModel, self).__init__()
        #self.bert = PostTrainedBert(load_post_trained_bert, post_trained_bert_file)
        self.rnn = layers.LSTM(768, dropout=dropout_rate)
        self.classifier_part1 = keras.models.Sequential()
        self.classifier_part1.add(keras.Input(shape=(768,)))
        self.classifier_part1.add(layers.Dense(384, activation='selu'))
        self.dropout = layers.Dropout(dropout_rate)
        self.classifier_part2 = keras.models.Sequential()
        self.classifier_part2.add(keras.Input(shape=(384,)))
        self.classifier_part2.add(layers.Dense(5, activation='softmax'))

    def call(self, inputs, sep_positions, training=False):
        embeddings = inputs[0]
        embedding = embeddings[0]
        sentence_embed = embedding[sep_positions[0]+1: sep_positions[1]]
        sentence_embed = sentence_embed[np.newaxis, :]
        rnn_embed = self.rnn(sentence_embed, training=training)
        classified_result = self.classifier_part1(rnn_embed)
        classified_result = self.dropout(classified_result, training=training)
        classified_result = self.classifier_part2(classified_result)
        return classified_result'''


def n_loss(result, label, class_frequency):
    LCE = tf.reduce_mean(-(label * tf.math.log(tf.clip_by_value(result, 1e-8, 1.0)) + (1 - label) * tf.math.log(tf.clip_by_value(1-result, 1e-8, 1.0))) * class_frequency) / len(
        class_frequency)
    return LCE


def focal_loss(result, label, alpha, gamma):
    a = alpha*label*((1-result)**gamma)*tf.math.log(tf.clip_by_value(result, 1e-8, 1.0)) + (1-alpha)*(1-label)*\
        (result**gamma)*tf.math.log(tf.clip_by_value(1-result, 1e-8, 1.0))
    LCE = tf.reduce_mean(-a)
    return LCE


def train():
    log_file = open("log.txt", "w+", encoding="utf8")
    #loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    #bert = PostTrainedBert(args.load_post_trained_bert, args.post_trained_bert_file)
    model = PersonaEmotionXModel(args.load_post_trained_bert, args.post_trained_bert_file, args.dropout, args.pooling_way, "emotion_classification")
    tokenizer = model.bert.tokenizer
    label_seq = load_label_seq("dialogue_label_seq.txt")
    '''data, frequency, tr_step, tr_nums = new_load_data(args.dataset, tokenizer, label_seq)
    #data, frequency, tr_step, tr_nums = new_load_data(args.dataset, tokenizer, label_seq)
    eval_data, _, eval_step, _ = new_load_data(args.evalset, tokenizer, label_seq)'''
    data, frequency, tr_step, tr_nums = persona_label_data("./dialogue_data/train_contexts.txt", "./dialogue_data/train_emotions.txt", "./dialogue_data/train_your_personas.txt", "./dialogue_data/train_partner_personas.txt", tokenizer, label_seq)
    eval_data, _, eval_step, _ = persona_label_data("./dialogue_data/test_contexts.txt", "./dialogue_data/test_emotions.txt", "./dialogue_data/test_your_personas.txt", "./dialogue_data/test_partner_personas.txt", tokenizer, label_seq)

    '''warm_up_steps = 2000
    global_step = len(data)*args.epoch-warm_up_steps
    first_decay_steps = 1000
    lr_decayed = tf.compat.v1.train.cosine_decay_restarts(
        args.max_lr, global_step, first_decay_steps, alpha=1e-7
    )'''
    optimizer = keras.optimizers.Adam(args.max_lr, epsilon=args.adam_epsilon, clipvalue=args.max_grad_norm)
    for i in range(args.epoch):
        total_loss = 0.
        print("Training for epoch {}".format(i))
        shuffle(data)
        train_true_num = 0.
        step = 0
        label_num = 0
        for d in data:
            x = tf.convert_to_tensor(np.array(d[0])[np.newaxis, :], dtype=np.int32)
            yp = tf.convert_to_tensor(np.array(d[2])[np.newaxis, :], dtype=np.int32)
            pp = tf.convert_to_tensor(np.array(d[4])[np.newaxis, :], dtype=np.int32)
            step += 1
            labels = []
            poss = []
            for y, s in d[1]:
                labels.append(y)
                poss.append(s)
            labels = tf.stack(labels, axis=0)
            with tf.GradientTape() as tape:
                predictions = model(x, poss, yp, d[3], pp, d[5], True)
                #loss_value = focal_loss(predictions, labels, 0.25, 2)
                loss_value = n_loss(predictions, labels, frequency)
            gradients = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))
            a = tf.argmax(predictions[:, :-1], 1)
            b = tf.argmax(labels, 1)
            for q in range(b.get_shape()[0]):
                if b[q] != 4:
                    if a[q] == b[q]:
                        train_true_num += 1
                    label_num += 1
            total_loss += loss_value
            if step % 1000 == 0:
                print("epoch: {} step: {} training loss: {}".format(i, step, total_loss / step))
        print("Train acc for epoch {} is {}".format(i, train_true_num / label_num))
        true_num = 0.
        compare_result = []
        test_step = 0
        for d in eval_data:
            x = tf.convert_to_tensor(np.array(d[0])[np.newaxis, :], dtype=np.int32)
            yp = tf.convert_to_tensor(np.array(d[2])[np.newaxis, :], dtype=np.int32)
            pp = tf.convert_to_tensor(np.array(d[4])[np.newaxis, :], dtype=np.int32)
            labels = []
            poss = []
            for y, s in d[1]:
                labels.append(y)
                poss.append(s)
            labels = tf.stack(labels, axis=0)
            result = model(x, poss, yp, d[3], pp, d[5], False)
            a = tf.argmax(result[:, :-1], 1)
            b = tf.argmax(labels, 1)
            for q in range(b.get_shape()[0]):
                if b[q] != 4:
                    if a[q] == b[q]:
                        true_num += 1
                    test_step += 1
                    k = [a[q], b[q]]
                    compare_result.append(k)
        print("Eval acc for epoch {} is {}".format(i, true_num/test_step))
        micro_f1, f1s = cal_score(compare_result, label_seq)
        print("micro_f1: {}".format(micro_f1))
        print(label_seq)
        print("F1s: {}".format(f1s))
        log_file.write("F1s: {}".format(f1s)+"\n")
        print("------------------------")
        os.makedirs(args.save_path+"checkpoint_{}/".format(i))
        model.save_weights(args.save_path+"checkpoint_{}/".format(i)+"checkpoint.h5")


def cal_score(results, label_seq):
    TPs = np.zeros([len(label_seq)-1])
    FPs = np.zeros([len(label_seq)-1])
    FNs = np.zeros([len(label_seq)-1])
    for b, a in results:
        '''a = tf.argmax(y)
        b = tf.argmax(result)'''
        if b == a:
            TPs[a] += 1
        else:
            FPs[a] += 1
            FNs[b] += 1
    '''TPs = TPs[0:-1]
    FPs = FPs[0:-1]
    FNs = FNs[0:-1]'''
    precisions = TPs/(TPs+FPs)
    recalls = TPs/(TPs+FNs)
    F1s = 2*precisions*recalls/(precisions+recalls)
    micro_p = np.sum(TPs)/np.sum(TPs+FPs)
    micro_r = np.sum(TPs)/np.sum(TPs+FNs)
    micro_f1 = 2*micro_p*micro_r/(micro_p+micro_r)
    return micro_f1, F1s


def test():
    #bert = PostTrainedBert(args.load_post_trained_bert, args.post_trained_bert_file)
    model = PosEmotionXModel(args.load_post_trained_bert, args.post_trained_bert_file, args.dropout, args.pooling_way, "emotion_classification")
    model.load_weights(args.save_path)
    tokenizer = model.bert.tokenizer
    label_seq = load_label_seq("dialogue_label_seq.txt")
    test_data, _, _, _ = load_label_data("./dialogue_data/test_contexts.txt", "./dialogue_data/test_emotions.txt", tokenizer, label_seq)
    true_num = 0.
    compare_result = []
    test_step = 0
    for d in test_data:
        x = tf.convert_to_tensor(np.array(d[0])[np.newaxis, :], dtype=np.int32)
        labels = []
        poss = []
        for y, s in d[1]:
            labels.append(y)
            poss.append(s)
        labels = tf.stack(labels, axis=0)
        result = model(x, poss, True)
        a = tf.argmax(result[:, :-1], 1)
        b = tf.argmax(labels, 1)
        for q in range(b.get_shape()[0]):
            if b[q] != 4:
                if a[q] == b[q]:
                    true_num += 1
                test_step += 1
                k = [a[q], b[q]]
                compare_result.append(k)
    print("Test acc is {}".format(true_num / test_step))
    micro_f1, f1s = cal_score(compare_result, label_seq)
    print("micro_f1: {}".format(micro_f1))
    print(label_seq)
    print("F1s: {}".format(f1s))
    print("------------------------")


if __name__=="__main__":
    '''pt = PostTrainedBert(False, None)
    a = pt.tokenizer.encode(["Hello", "are", "you", "people", "?"])
    a = np.array(a)
    d = a[np.newaxis, :]
    print(pt(d, None)[0])
    a = EmotionXModel(False, None, 0.2, "max")
    input = np.array([[101, 2115, 2171, 1010, 3531, 1029, 102, 9558, 100, 1012, 102, 8529, 1011, 17012, 1010, 1998, 2106, 2017, 2994, 2039, 2035, 2305, 1999, 7547, 2005, 2115, 3637, 2817, 1012, 7910, 1010, 2909, 1029, 102, 2748, 2002, 2106, 1012, 102, 100, 2157, 1010, 2057, 100, 2655, 2017, 1999, 1037, 2261, 2781, 1012, 102, 4931, 1010, 4638, 2041, 2008, 2611, 999, 2016, 2003, 2428, 2980, 999, 102, 3398, 1010, 2016, 2003, 1012, 10166, 999, 2129, 2017, 24341, 1005, 1029, 102, 2054, 100, 102, 2017, 100, 2746, 2006, 2000, 1996, 2972, 2282, 999, 1045, 100, 13814, 1012, 102, 1045, 100, 21562, 1012, 102, 7632, 1012, 102, 7632, 1012, 102, 2017, 2568, 2065, 1045, 2133, 102, 2053, 1010, 3531, 1012, 102, 2061, 7910, 1010, 2054, 2024, 2017, 1999, 2005, 1029, 102, 1045, 2831, 1999, 2026, 3637, 1012, 102, 2054, 1037, 16507, 1010, 1045, 4952, 1999, 2026, 3637, 1012, 102, 2061, 2339, 2079, 100, 2017, 2507, 2033, 2115, 2193, 1029, 102]])
    place = [[0, 6, 10, 33, 38, 51, 64, 77, 80, 94, 99, 102, 105, 111, 116, 126, 133, 144, 155]]
    b = a(input, place, None)
    print(loss([tf.constant([[0.1,0.3,0.5,0.1],[0.1,0.3,0.5,0.1]])],
               [tf.constant([[1.,0,0,0],[0,1,0,0]])], np.array([[0.1,0.3,0.5,0.1]])))'''
    if args.mode == "train":
        train()
    else:
        test()
    '''result = np.array([0., 1., 0])
    label = np.array([0.1, 0.8, 0.1])
    print(focal_loss(result, label, 0.25, 2))'''
