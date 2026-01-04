from __future__ import print_function
import time
import sys
import argparse
import random
import torch
import gc
import torch.optim as optim
import numpy as np
from utils.metric import get_ner_fmeasure
from model.seqlabel import SeqLabel
from model.sentclassifier import SentClassifier
from utils.data import Data

try:
    import cPickle as pickle
except ImportError:
    import pickle

seed_num = 42
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)

def data_initialization(data):
    data.initial_feature_alphabets()
    data.build_alphabet(data.train_dir)
    data.build_alphabet(data.dev_dir)
    data.build_alphabet(data.test_dir)
    data.fix_alphabet()

def predict_check(pred_variable, gold_variable, mask_variable, sentence_classification=False):
    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlaped = (pred == gold)
    if sentence_classification:
        right_token = np.sum(overlaped)
        total_token = overlaped.shape[0]
    else:
        right_token = np.sum(overlaped * mask)
        total_token = mask.sum()
    return right_token, total_token

def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover, sentence_classification=False):
    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    if sentence_classification:
        pred_tag = pred_variable.cpu().data.numpy().tolist()
        gold_tag = gold_variable.cpu().data.numpy().tolist()
        pred_label = [label_alphabet.get_instance(pred) for pred in pred_tag]
        gold_label = [label_alphabet.get_instance(gold) for gold in gold_tag]
    else:
        seq_len = gold_variable.size(1)
        mask = mask_variable.cpu().data.numpy()
        pred_tag = pred_variable.cpu().data.numpy()
        gold_tag = gold_variable.cpu().data.numpy()
        batch_size = mask.shape[0]
        pred_label, gold_label = [], []
        for idx in range(batch_size):
            pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
            gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
            assert(len(pred) == len(gold))
            pred_label.append(pred)
            gold_label.append(gold)
    return pred_label, gold_label

def recover_nbest_label(pred_variable, mask_variable, label_alphabet, word_recover):
    pred_variable = pred_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    seq_len = pred_variable.size(1)
    nbest = pred_variable.size(2)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    for idx in range(batch_size):
        pred = []
        for idz in range(nbest):
            each_pred = [label_alphabet.get_instance(pred_tag[idx][idy][idz]) for idy in range(seq_len) if mask[idx][idy] != 0]
            pred.append(each_pred)
        pred_label.append(pred)
    return pred_label

def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr / (1 + decay_rate * epoch)
    print(" Learning rate is set as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def evaluate(data, model, name, nbest=None):
    if name == "train":
        instances = data.train_Ids
    elif name == "dev":
        instances = data.dev_Ids
    elif name == 'test':
        instances = data.test_Ids
    elif name == 'raw':
        instances = data.raw_Ids
    else:
        print("Error: wrong evaluate name,", name)
        exit(1)

    model.eval()
    batch_size = data.HP_batch_size
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num // batch_size + 1

    nbest_pred_results, pred_scores = [], []
    pred_results, gold_results = [], []

    # optional fusion token accuracy
    fus_right, fus_total = 0, 0

    for batch_id in range(total_batch):
        start = batch_id * batch_size
        end = min((batch_id + 1) * batch_size, train_num)
        instance = instances[start:end]
        if not instance:
            continue

        (batch_word, batch_features, batch_wordlen, batch_wordrecover,
         batch_char, batch_charlen, batch_charrecover,
         batch_label_lid, batch_label_fus, mask) = batchify_with_label(instance, data.HP_gpu, False, data.sentence_classification)

        if nbest and not data.sentence_classification:
            scores, nbest_tag_seq = model.decode_nbest(batch_word, batch_features, batch_wordlen,
                                                       batch_char, batch_charlen, batch_charrecover,
                                                       mask, nbest)
            nbest_pred_result = recover_nbest_label(nbest_tag_seq, mask, data.label_alphabet, batch_wordrecover)
            nbest_pred_results += nbest_pred_result
            pred_scores += scores[batch_wordrecover].cpu().data.numpy().tolist()
            tag_seq = nbest_tag_seq[:, :, 0]
        else:
            tag_seq = model(batch_word, batch_features, batch_wordlen,
                            batch_char, batch_charlen, batch_charrecover, mask)

        pred_label, gold_label = recover_label(tag_seq, batch_label_lid, mask, data.label_alphabet, batch_wordrecover, data.sentence_classification)
        pred_results += pred_label
        gold_results += gold_label

        # Fusion accuracy (simple): compare gold to argmax of auxiliary head not exposed in forward.
        # We skip here to keep evaluation aligned with NCRF++ standard.
        # If you want fus acc during eval, we can add a model method to output fus predictions too.

    decode_time = time.time() - start_time
    speed = len(instances) / decode_time
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)

    if nbest and not data.sentence_classification:
        return speed, acc, p, r, f, nbest_pred_results, pred_scores

    return speed, acc, p, r, f, pred_results, pred_scores

def batchify_with_label(input_batch_list, gpu, if_train=True, sentence_classification=False):
    if sentence_classification:
        return batchify_sentence_classification_with_label(input_batch_list, gpu, if_train)
    else:
        return batchify_sequence_labeling_with_label(input_batch_list, gpu, if_train)

def batchify_sequence_labeling_with_label(input_batch_list, gpu, if_train=True):
    """
    Now each instance is:
      [word_ids, feature_ids, char_ids, lid_label_ids, fusion_label_ids]
    """
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    features = [np.asarray(sent[1]) for sent in input_batch_list]
    feature_num = len(features[0][0])
    chars = [sent[2] for sent in input_batch_list]
    lid_labels = [sent[3] for sent in input_batch_list]
    fus_labels = [sent[4] for sent in input_batch_list]

    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max().item()

    word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
    lid_label_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
    fus_label_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()

    feature_seq_tensors = [torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
                           for _ in range(feature_num)]
    mask = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).bool()

    for idx, (seq, lid_lab, fus_lab, seqlen) in enumerate(zip(words, lid_labels, fus_labels, word_seq_lengths)):
        seqlen = seqlen.item()
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        lid_label_seq_tensor[idx, :seqlen] = torch.LongTensor(lid_lab)
        fus_label_seq_tensor[idx, :seqlen] = torch.LongTensor(fus_lab)
        mask[idx, :seqlen] = torch.Tensor([1] * seqlen)
        for idy in range(feature_num):
            feature_seq_tensors[idy][idx, :seqlen] = torch.LongTensor(features[idx][:, idy])

    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    lid_label_seq_tensor = lid_label_seq_tensor[word_perm_idx]
    fus_label_seq_tensor = fus_label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    for idx in range(feature_num):
        feature_seq_tensors[idx] = feature_seq_tensors[idx][word_perm_idx]

    # chars
    pad_chars = [chars[idx] + [[0]] * (max_seq_len - len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_word_len), requires_grad=if_train).long()
    char_seq_lengths = torch.LongTensor(length_list)

    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size * max_seq_len, -1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size * max_seq_len,)
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)

    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        lid_label_seq_tensor = lid_label_seq_tensor.cuda()
        fus_label_seq_tensor = fus_label_seq_tensor.cuda()
        for idx in range(feature_num):
            feature_seq_tensors[idx] = feature_seq_tensors[idx].cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
        mask = mask.cuda()

    return (word_seq_tensor, feature_seq_tensors, word_seq_lengths, word_seq_recover,
            char_seq_tensor, char_seq_lengths, char_seq_recover,
            lid_label_seq_tensor, fus_label_seq_tensor, mask)

def batchify_sentence_classification_with_label(input_batch_list, gpu, if_train=True):
    # unchanged original (not used for two-task)
    raise NotImplementedError("Two-task setup is for sequence labeling only.")

def train(data):
    print("Training model...")
    data.show_data_summary()
    save_data_name = data.model_dir + ".dset"
    data.save(save_data_name)

    if data.sentence_classification:
        model = SentClassifier(data)
    else:
        model = SeqLabel(data)

    if data.optimizer.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=data.HP_lr, momentum=data.HP_momentum, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    else:
        print("Optimizer illegal: %s" % (data.optimizer))
        exit(1)

    best_dev = -10

    for idx in range(data.HP_iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        print("Epoch: %s/%s" % (idx, data.HP_iteration))
        if data.optimizer == "SGD":
            optimizer = lr_decay(optimizer, idx, data.HP_lr_decay, data.HP_lr)

        sample_loss, total_loss = 0, 0
        right_token, whole_token = 0, 0

        random.shuffle(data.train_Ids)
        print("Shuffle: first input word list:", data.train_Ids[0][0])

        model.train()
        model.zero_grad()
        batch_size = data.HP_batch_size
        train_num = len(data.train_Ids)
        total_batch = train_num // batch_size + 1

        for batch_id in range(total_batch):
            start = batch_id * batch_size
            end = min((batch_id + 1) * batch_size, train_num)
            instance = data.train_Ids[start:end]
            if not instance:
                continue

            (batch_word, batch_features, batch_wordlen, batch_wordrecover,
             batch_char, batch_charlen, batch_charrecover,
             batch_label_lid, batch_label_fus, mask) = batchify_with_label(instance, data.HP_gpu, True, data.sentence_classification)

            loss, tag_seq = model.calculate_loss(batch_word, batch_features, batch_wordlen,
                                                 batch_char, batch_charlen, batch_charrecover,
                                                 batch_label_lid, batch_label_fus, mask)

            right, whole = predict_check(tag_seq, batch_label_lid, mask, data.sentence_classification)
            right_token += right
            whole_token += whole

            sample_loss += loss.item()
            total_loss += loss.item()

            if end % 500 == 0:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" %
                      (end, temp_cost, sample_loss, right_token, whole_token, (right_token + 0.) / whole_token))
                if sample_loss > 1e8 or str(sample_loss) == "nan":
                    print("ERROR: LOSS EXPLOSION (>1e8) ! EXIT....")
                    exit(1)
                sys.stdout.flush()
                sample_loss = 0

            loss.backward()
            optimizer.step()
            model.zero_grad()

        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s" %
              (idx, epoch_cost, train_num / epoch_cost, total_loss))

        if total_loss > 1e8 or str(total_loss) == "nan":
            print("ERROR: LOSS EXPLOSION (>1e8) ! EXIT....")
            exit(1)

        speed, acc, p, r, f, _, _ = evaluate(data, model, "dev")
        if data.seg:
            current_score = f
            print("Dev: speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (speed, acc, p, r, f))
        else:
            current_score = acc
            print("Dev: speed: %.2fst/s; acc: %.4f" % (speed, acc))

        if current_score > best_dev:
            print("Exceed previous best score:", best_dev)
            model_name = data.model_dir + '.' + str(idx) + ".model"
            print("Save current best model in file:", model_name)
            torch.save(model.state_dict(), model_name)
            best_dev = current_score

        speed, acc, p, r, f, _, _ = evaluate(data, model, "test")
        if data.seg:
            print("Test: speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (speed, acc, p, r, f))
        else:
            print("Test: speed: %.2fst/s; acc: %.4f" % (speed, acc))

        gc.collect()

def load_model_decode(data, name):
    print("Load Model from file: ", data.model_dir)
    if data.sentence_classification:
        model = SentClassifier(data)
    else:
        model = SeqLabel(data)
    model.load_state_dict(torch.load(data.load_model_dir))

    print("Decode %s data, nbest: %s ..." % (name, data.nbest))
    start_time = time.time()
    speed, acc, p, r, f, pred_results, pred_scores = evaluate(data, model, name, data.nbest)
    time_cost = time.time() - start_time
    if data.seg:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" %
              (name, time_cost, speed, acc, p, r, f))
    else:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f" % (name, time_cost, speed, acc))
    return pred_results, pred_scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Two-task NCRF++ fork: LID + Fusion')
    parser.add_argument('--config', help='Configuration File', default='None')
    args = parser.parse_args()

    data = Data()
    data.HP_gpu = torch.cuda.is_available()
    data.read_config(args.config)

    status = data.status.lower()
    print("Seed num:", seed_num)

    if status == 'train':
        print("MODEL: train")
        data_initialization(data)
        data.generate_instance('train')
        data.generate_instance('dev')
        data.generate_instance('test')
        data.build_pretrain_emb()
        train(data)
    elif status == 'decode':
        print("MODEL: decode")
        data.load(data.dset_dir)
        data.read_config(args.config)
        data.show_data_summary()
        data.generate_instance('raw')
        decode_results, pred_scores = load_model_decode(data, 'raw')
        #if data.nbest and not data.sentence_classification:
        #    data.write_nbest_decoded_results(decode_results, pred_scores, 'raw')
        #else:
         #   data.write_decoded_results(decode_results, 'raw')
        #data.write_decoded_results(decode_results, 'raw')
        # --- ensure writer receives BOTH tasks if available ---
        if isinstance(decode_results, tuple) and len(decode_results) == 2:
            lid_decode_results, fus_decode_results = decode_results
            decode_results = {"lid": lid_decode_results, "fusion": fus_decode_results}

        elif isinstance(decode_results, dict):
            # already good: expects keys "lid" and "fusion"
            pass

        else:
            # LID-only decode; fusion predictions not produced
            decode_results = {"lid": decode_results, "fusion": None}

        data.write_decoded_results(decode_results, 'raw')

    else:
        print("Invalid argument! Please use valid arguments! (train/decode)")
