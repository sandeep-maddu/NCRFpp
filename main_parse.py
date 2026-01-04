# -*- coding: utf-8 -*-
# Two-task NCRF++ runner (LID + Fusion) with shared encoder + 2 CRF heads
# Updated from legacy main_parse.py to support TWO label columns per token line:
#   ... <LID_LABEL> <FUSION_LABEL>
# Example:
#   collegeki ... B-EN B-FU
#   trainlo   ... B-EN B-FU
#   mundu     ... B-TE B-NF
#
# Assumptions about your forked model/seqlabel.py:
#   - model.calculate_loss(..., lid_labels, fusion_labels, mask) -> (loss, lid_tag_seq, fusion_tag_seq)
#   - model.forward(..., mask) -> (lid_tag_seq, fusion_tag_seq)
#   - model.decode_nbest(..., nbest) -> (lid_scores, lid_nbest_tag_seq, fusion_scores, fusion_nbest_tag_seq) [optional]
#     If you did NOT implement nbest for fusion, we fall back to greedy decode for fusion.

from __future__ import print_function
import time
import sys
import argparse
import random
import torch
import gc
import numpy as np

from utils.metric import get_ner_fmeasure
from model.seqlabel import SeqLabel
from utils.data import Data

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


def predict_check(pred_variable, gold_variable, mask_variable):
    """
    pred_variable/gold_variable: (batch_size, sent_len)
    mask_variable: (batch_size, sent_len) bool
    """
    pred = pred_variable.detach().cpu().numpy()
    gold = gold_variable.detach().cpu().numpy()
    mask = mask_variable.detach().cpu().numpy().astype(bool)
    overlapped = (pred == gold)
    right_token = np.sum(overlapped & mask)
    total_token = np.sum(mask)
    return right_token, total_token


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover):
    """
    pred_variable/gold_variable: (batch_size, sent_len)
    mask_variable: (batch_size, sent_len) bool
    """
    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]

    mask = mask_variable.detach().cpu().numpy().astype(bool)
    pred_tag = pred_variable.detach().cpu().numpy()
    gold_tag = gold_variable.detach().cpu().numpy()

    batch_size, seq_len = pred_tag.shape
    pred_label, gold_label = [], []
    for i in range(batch_size):
        pred_seq = [label_alphabet.get_instance(int(pred_tag[i][j])) for j in range(seq_len) if mask[i][j]]
        gold_seq = [label_alphabet.get_instance(int(gold_tag[i][j])) for j in range(seq_len) if mask[i][j]]
        assert len(pred_seq) == len(gold_seq)
        pred_label.append(pred_seq)
        gold_label.append(gold_seq)
    return pred_label, gold_label


def recover_nbest_label(pred_variable, mask_variable, label_alphabet, word_recover):
    """
    pred_variable: (batch_size, sent_len, nbest)
    mask_variable: (batch_size, sent_len) bool
    """
    pred_variable = pred_variable[word_recover]
    mask_variable = mask_variable[word_recover]

    mask = mask_variable.detach().cpu().numpy().astype(bool)
    pred_tag = pred_variable.detach().cpu().numpy()

    batch_size, seq_len, nbest = pred_tag.shape
    out = []
    for i in range(batch_size):
        sent_nbest = []
        for k in range(nbest):
            seq_k = [label_alphabet.get_instance(int(pred_tag[i][j][k])) for j in range(seq_len) if mask[i][j]]
            sent_nbest.append(seq_k)
        out.append(sent_nbest)
    return out


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr / (1.0 + decay_rate * epoch)
    print(" Learning rate is set as:", lr)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return optimizer


def fusion_token_f1(gold_seqs, pred_seqs, positive_tags=("B-FU", "I-FU", "FU")):
    """
    gold_seqs/pred_seqs: list[list[str]] aligned token sequences (after masking)
    Computes micro P/R/F1 over tokens treating positive if tag in positive_tags (or endswith '-FU').
    """
    tp = fp = fn = 0
    pos_set = set([t.upper() for t in positive_tags])

    def is_pos(tag):
        u = tag.upper()
        if u in pos_set:
            return True
        # robust for BIO like B-FU / I-FU
        if u.endswith("-FU") or u.endswith("FU"):
            if "NF" in u:  # avoid B-NF false match
                return False
            return True
        return False

    for gsent, psent in zip(gold_seqs, pred_seqs):
        for g, p in zip(gsent, psent):
            gpos = is_pos(g)
            ppos = is_pos(p)
            if ppos and gpos:
                tp += 1
            elif ppos and (not gpos):
                fp += 1
            elif (not ppos) and gpos:
                fn += 1

    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return prec, rec, f1, tp, fp, fn


def evaluate(data, model, name, nbest=None):
    if name == "train":
        instances = data.train_Ids
    elif name == "dev":
        instances = data.dev_Ids
    elif name == "test":
        instances = data.test_Ids
    elif name == "raw":
        instances = data.raw_Ids
    else:
        print("Error: wrong evaluate name,", name)
        return None

    model.eval()
    batch_size = data.HP_batch_size
    start_time = time.time()

    pred_lid_results, gold_lid_results = [], []
    pred_fus_results, gold_fus_results = [], []

    nbest_lid_pred_results = []
    nbest_lid_scores = []

    total = len(instances)
    total_batch = total // batch_size + 1

    with torch.no_grad():
        for batch_id in range(total_batch):
            start = batch_id * batch_size
            end = min((batch_id + 1) * batch_size, total)
            instance = instances[start:end]
            if not instance:
                continue

            (batch_word, batch_features, batch_wordlen, batch_wordrecover,
             batch_char, batch_charlen, batch_charrecover,
             batch_lid_label, batch_fus_label, mask) = batchify_with_label(instance, data.HP_gpu)

            if nbest:
                # If your SeqLabel.decode_nbest returns only LID nbest, we handle both possibilities.
                try:
                    out = model.decode_nbest(batch_word, batch_features, batch_wordlen,
                                            batch_char, batch_charlen, batch_charrecover,
                                            mask, nbest)
                    if len(out) == 4:
                        lid_scores, lid_nbest_tag_seq, fus_scores, fus_nbest_tag_seq = out
                    else:
                        lid_scores, lid_nbest_tag_seq = out
                        fus_scores, fus_nbest_tag_seq = None, None
                except Exception:
                    lid_scores, lid_nbest_tag_seq = model.decode_nbest(batch_word, batch_features, batch_wordlen,
                                                                      batch_char, batch_charlen, batch_charrecover,
                                                                      mask, nbest)
                    fus_scores, fus_nbest_tag_seq = None, None

                nbest_pred_lid = recover_nbest_label(lid_nbest_tag_seq, mask, data.label_alphabet, batch_wordrecover)
                nbest_lid_pred_results += nbest_pred_lid
                nbest_lid_scores += lid_scores[batch_wordrecover].detach().cpu().numpy().tolist()

                lid_tag_seq = lid_nbest_tag_seq[:, :, 0]
                if fus_nbest_tag_seq is not None:
                    fus_tag_seq = fus_nbest_tag_seq[:, :, 0]
                else:
                    # fallback greedy fusion decode
                    lid_tag_seq_tmp, fus_tag_seq = model(batch_word, batch_features, batch_wordlen,
                                                        batch_char, batch_charlen, batch_charrecover, mask)
            else:
                lid_tag_seq, fus_tag_seq = model(batch_word, batch_features, batch_wordlen,
                                                 batch_char, batch_charlen, batch_charrecover, mask)

            pred_lid, gold_lid = recover_label(lid_tag_seq, batch_lid_label, mask, data.label_alphabet, batch_wordrecover)
            pred_fus, gold_fus = recover_label(fus_tag_seq, batch_fus_label, mask, data.fusion_label_alphabet, batch_wordrecover)

            pred_lid_results += pred_lid
            gold_lid_results += gold_lid
            pred_fus_results += pred_fus
            gold_fus_results += gold_fus

    decode_time = time.time() - start_time
    speed = len(instances) / (decode_time + 1e-12)

    # LID metrics (sequence labeling style)
    lid_acc, lid_p, lid_r, lid_f = get_ner_fmeasure(gold_lid_results, pred_lid_results, data.tagScheme)

    # Fusion metrics (token-level F1 on positives)
    fus_p, fus_r, fus_f, tp, fp, fn = fusion_token_f1(gold_fus_results, pred_fus_results, positive_tags=("B-FU", "I-FU", "FU"))

    if nbest:
        return (speed, lid_acc, lid_p, lid_r, lid_f,
                fus_p, fus_r, fus_f,
                nbest_lid_pred_results, nbest_lid_scores,
                pred_lid_results, pred_fus_results)
    return (speed, lid_acc, lid_p, lid_r, lid_f,
            fus_p, fus_r, fus_f,
            pred_lid_results, pred_fus_results)


def batchify_with_label(input_batch_list, gpu):
    """
    input_batch_list elements:
      [word_ids, feature_ids, char_ids, lid_label_ids, fusion_label_ids]

    Returns:
      word_seq_tensor, feature_seq_tensors, word_seq_lengths, word_seq_recover,
      char_seq_tensor, char_seq_lengths, char_seq_recover,
      lid_label_seq_tensor, fusion_label_seq_tensor, mask
    """
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    features = [np.asarray(sent[1]) for sent in input_batch_list]
    feature_num = len(features[0][0])
    chars = [sent[2] for sent in input_batch_list]
    lid_labels = [sent[3] for sent in input_batch_list]
    fus_labels = [sent[4] for sent in input_batch_list]

    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = int(word_seq_lengths.max().item())

    word_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    lid_label_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    fus_label_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)

    feature_seq_tensors = [torch.zeros((batch_size, max_seq_len), dtype=torch.long) for _ in range(feature_num)]
    mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)

    for i, (seq, lid_lab, fus_lab, seqlen) in enumerate(zip(words, lid_labels, fus_labels, word_seq_lengths)):
        L = int(seqlen.item())
        word_seq_tensor[i, :L] = torch.LongTensor(seq)
        lid_label_seq_tensor[i, :L] = torch.LongTensor(lid_lab)
        fus_label_seq_tensor[i, :L] = torch.LongTensor(fus_lab)
        mask[i, :L] = True
        for f in range(feature_num):
            feature_seq_tensors[f][i, :L] = torch.LongTensor(features[i][:, f])

    # sort by length
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    lid_label_seq_tensor = lid_label_seq_tensor[word_perm_idx]
    fus_label_seq_tensor = fus_label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    for f in range(feature_num):
        feature_seq_tensors[f] = feature_seq_tensors[f][word_perm_idx]

    # char padding
    pad_chars = [chars[i] + [[0]] * (max_seq_len - len(chars[i])) for i in range(batch_size)]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = int(max(map(max, length_list))) if max_seq_len > 0 else 1

    char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_word_len), dtype=torch.long)
    char_seq_lengths = torch.LongTensor(length_list)

    for i, (seq_chars, seq_lens) in enumerate(zip(pad_chars, char_seq_lengths)):
        for j, (wchars, wlen) in enumerate(zip(seq_chars, seq_lens)):
            wlen = int(wlen.item())
            if wlen > 0:
                char_seq_tensor[i, j, :wlen] = torch.LongTensor(wchars)

    # reshape and sort chars
    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size * max_seq_len, -1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size * max_seq_len)
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)

    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        lid_label_seq_tensor = lid_label_seq_tensor.cuda()
        fus_label_seq_tensor = fus_label_seq_tensor.cuda()
        mask = mask.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_lengths = char_seq_lengths.cuda()
        char_seq_recover = char_seq_recover.cuda()
        for f in range(feature_num):
            feature_seq_tensors[f] = feature_seq_tensors[f].cuda()

    return (word_seq_tensor, feature_seq_tensors, word_seq_lengths, word_seq_recover,
            char_seq_tensor, char_seq_lengths, char_seq_recover,
            lid_label_seq_tensor, fus_label_seq_tensor, mask)


def train(data):
    print("Training model (two-task: LID + Fusion)...")
    data.show_data_summary()
    save_data_name = data.model_dir + ".dset"
    data.save(save_data_name)

    model = SeqLabel(data)

    if data.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=data.HP_lr, momentum=data.HP_momentum, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    else:
        print("Optimizer illegal:", data.optimizer)
        exit(1)

    best_dev = -1e9

    for epoch in range(data.HP_iteration):
        epoch_start = time.time()
        print("Epoch: %s/%s" % (epoch, data.HP_iteration))
        if data.optimizer.lower() == "sgd":
            optimizer = lr_decay(optimizer, epoch, data.HP_lr_decay, data.HP_lr)

        random.shuffle(data.train_Ids)
        model.train()
        model.zero_grad()

        batch_size = data.HP_batch_size
        train_num = len(data.train_Ids)
        total_batch = train_num // batch_size + 1

        running_loss = 0.0
        right_lid = total_lid = 0
        right_fus = total_fus = 0

        for batch_id in range(total_batch):
            start = batch_id * batch_size
            end = min((batch_id + 1) * batch_size, train_num)
            instance = data.train_Ids[start:end]
            if not instance:
                continue

            (batch_word, batch_features, batch_wordlen, batch_wordrecover,
             batch_char, batch_charlen, batch_charrecover,
             batch_lid_label, batch_fus_label, mask) = batchify_with_label(instance, data.HP_gpu)

            loss, lid_tag_seq, fus_tag_seq = model.calculate_loss(
                batch_word, batch_features, batch_wordlen,
                batch_char, batch_charlen, batch_charrecover,
                batch_lid_label, batch_fus_label, mask
            )

            loss.backward()
            optimizer.step()
            model.zero_grad()

            running_loss += float(loss.item())

            r1, t1 = predict_check(lid_tag_seq, batch_lid_label, mask)
            r2, t2 = predict_check(fus_tag_seq, batch_fus_label, mask)
            right_lid += r1; total_lid += t1
            right_fus += r2; total_fus += t2

            if end % 500 == 0:
                print("  Instance: %s; loss: %.4f; LID acc: %.4f; FUS acc: %.4f" % (
                    end, running_loss,
                    (right_lid + 0.0) / max(total_lid, 1),
                    (right_fus + 0.0) / max(total_fus, 1)
                ))
                sys.stdout.flush()
                running_loss = 0.0

        epoch_cost = time.time() - epoch_start
        print("Epoch %d finished. Time: %.2fs, speed: %.2fst/s" % (epoch, epoch_cost, train_num / (epoch_cost + 1e-12)))

        # DEV
        out = evaluate(data, model, "dev", data.nbest)
        if data.nbest:
            (speed, lid_acc, lid_p, lid_r, lid_f,
             fus_p, fus_r, fus_f,
             _, _, _, _) = out
        else:
            (speed, lid_acc, lid_p, lid_r, lid_f,
             fus_p, fus_r, fus_f,
             _, _) = out

        # Choose selection score (keep LID F1 as primary, fusion as secondary)
        current_score = lid_f + 0.05 * fus_f

        print("Dev: speed: %.2fst/s; LID acc: %.4f p/r/f: %.4f/%.4f/%.4f | Fusion p/r/f: %.4f/%.4f/%.4f" % (
            speed, lid_acc, lid_p, lid_r, lid_f, fus_p, fus_r, fus_f
        ))

        if current_score > best_dev:
            print("Exceed previous best score:", best_dev)
            model_name = data.model_dir + "." + str(epoch) + ".model"
            print("Save current best model:", model_name)
            torch.save(model.state_dict(), model_name)
            best_dev = current_score

        # TEST each epoch (optional but kept to match original behavior)
        out = evaluate(data, model, "test", data.nbest)
        if data.nbest:
            (speed, lid_acc, lid_p, lid_r, lid_f,
             fus_p, fus_r, fus_f,
             _, _, _, _) = out
        else:
            (speed, lid_acc, lid_p, lid_r, lid_f,
             fus_p, fus_r, fus_f,
             _, _) = out

        print("Test: speed: %.2fst/s; LID acc: %.4f p/r/f: %.4f/%.4f/%.4f | Fusion p/r/f: %.4f/%.4f/%.4f" % (
            speed, lid_acc, lid_p, lid_r, lid_f, fus_p, fus_r, fus_f
        ))
        gc.collect()


def write_decode_two_labels(words_text, lid_preds, fus_preds, out_path):
    """
    Writes token + LID + FUS columns, sentence separated by blank line.
    words_text: list[list[str]] original tokens
    lid_preds/fus_preds: list[list[str]] same lengths
    """
    with open(out_path, "w", encoding="utf8") as f:
        for sent_words, sent_lid, sent_fus in zip(words_text, lid_preds, fus_preds):
            for w, l1, l2 in zip(sent_words, sent_lid, sent_fus):
                f.write("%s %s %s\n" % (w, l1, l2))
            f.write("\n")


def load_model_decode(data, name):
    print("Load Model from file:", data.model_dir)
    model = SeqLabel(data)
    model.load_state_dict(torch.load(data.load_model_dir, map_location=None))

    print("Decode %s data, nbest: %s ..." % (name, data.nbest))
    out = evaluate(data, model, name, data.nbest)

    # unpack
    if data.nbest:
        (speed, lid_acc, lid_p, lid_r, lid_f,
         fus_p, fus_r, fus_f,
         _, _, pred_lid_results, pred_fus_results) = out
    else:
        (speed, lid_acc, lid_p, lid_r, lid_f,
         fus_p, fus_r, fus_f,
         pred_lid_results, pred_fus_results) = out

    print("%s: speed: %.2fst/s; LID acc: %.4f p/r/f: %.4f/%.4f/%.4f | Fusion p/r/f: %.4f/%.4f/%.4f" % (
        name, speed, lid_acc, lid_p, lid_r, lid_f, fus_p, fus_r, fus_f
    ))
    return pred_lid_results, pred_fus_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tuning with NCRF++ (two-task: LID + Fusion)")
    parser.add_argument("--wordemb", help="Embedding for words", default="None")
    parser.add_argument("--charemb", help="Embedding for chars", default="None")
    parser.add_argument("--status", choices=["train", "decode"], default="train")
    parser.add_argument("--savemodel", default="data/model/saved_model.lstmcrf.")
    parser.add_argument("--savedset", help="Dir of saved data setting")
    parser.add_argument("--train", default="data/conll03/train.bmes")
    parser.add_argument("--dev", default="data/conll03/dev.bmes")
    parser.add_argument("--test", default="data/conll03/test.bmes")
    parser.add_argument("--seg", default="True")
    parser.add_argument("--raw")
    parser.add_argument("--loadmodel")
    parser.add_argument("--output")
    args = parser.parse_args()

    data = Data()
    data.train_dir = args.train
    data.dev_dir = args.dev
    data.test_dir = args.test
    data.model_dir = args.savemodel
    data.dset_dir = args.savedset
    data.HP_gpu = torch.cuda.is_available()
    print("Seed num:", seed_num)

    # word emb path
    if args.wordemb != "None":
        data.word_emb_dir = args.wordemb
    else:
        # legacy default
        data.word_emb_dir = "../data/glove.6B.100d.txt"

    status = args.status.lower()

    if status == "train":
        print("MODEL: train")
        data_initialization(data)
        data.use_char = True
        data.HP_batch_size = 10
        data.HP_lr = 0.015
        # NOTE: in Data() it is "char_feature_extractor", but your legacy script used "char_seq_feature".
        # Keeping original intent: CNN chars
        data.char_feature_extractor = "CNN"

        data.generate_instance("train")
        data.generate_instance("dev")
        data.generate_instance("test")
        data.build_pretrain_emb()
        train(data)

    elif status == "decode":
        print("MODEL: decode")
        data.load(data.dset_dir)
        data.raw_dir = args.raw
        data.decode_dir = args.output
        data.load_model_dir = args.loadmodel

        data.show_data_summary()
        data.generate_instance("raw")
        print("nbest:", data.nbest)

        lid_preds, fus_preds = load_model_decode(data, "raw")

        # Write 3-column output: token LID FUS
        # data.raw_texts format: [words, features, chars, labels] in original Data,
        # but for our fork it should still keep words in raw_texts[idx][0]
        raw_words = [x[0] for x in data.raw_texts]
        write_decode_two_labels(raw_words, lid_preds, fus_preds, data.decode_dir)

        print("Wrote decoded output with 2 labels to:", data.decode_dir)

    else:
        print("Invalid argument! Please use valid arguments! (train/decode)")
