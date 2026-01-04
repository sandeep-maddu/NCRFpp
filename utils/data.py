from __future__ import print_function
from __future__ import absolute_import
import sys
from .alphabet import Alphabet
from .functions import *

try:
    import cPickle as pickle
except ImportError:
    import pickle as pickle

START = "</s>"
UNKNOWN = "</unk>"
PADDING = "</pad>"

class Data:
    def __init__(self):
        self.sentence_classification = False
        self.MAX_SENTENCE_LENGTH = 250
        self.MAX_WORD_LENGTH = -1
        self.number_normalized = True
        self.norm_word_emb = False
        self.norm_char_emb = False
        self.word_alphabet = Alphabet('word')
        self.char_alphabet = Alphabet('character')

        self.feature_name = []
        self.feature_alphabets = []
        self.feature_num = len(self.feature_alphabets)
        self.feat_config = None

        # Main label alphabet = LID
        self.label_alphabet = Alphabet('label', True)
        self.tagScheme = "NoSeg"
        self.split_token = ' ||| '
        self.seg = True

        # NEW: Fusion label alphabet
        self.fusion_label_alphabet = Alphabet('fusion_label', True)
        self.fusion_label_alphabet_size = 0

        # NEW: two-task loss params
        self.lambda_fusion = 0.2           # default
        self.fusion_pos_weight = 10.0      # weight for FUS class vs NF (reviewer-safe default)

        ### I/O
        self.train_dir = None
        self.dev_dir = None
        self.test_dir = None
        self.raw_dir = None

        self.decode_dir = None
        self.dset_dir = None
        self.model_dir = None
        self.load_model_dir = None

        self.word_emb_dir = None
        self.char_emb_dir = None
        self.feature_emb_dirs = []

        self.train_texts = []
        self.dev_texts = []
        self.test_texts = []
        self.raw_texts = []

        self.train_Ids = []
        self.dev_Ids = []
        self.test_Ids = []
        self.raw_Ids = []

        self.pretrain_word_embedding = None
        self.pretrain_char_embedding = None
        self.pretrain_feature_embeddings = []

        self.label_size = 0
        self.word_alphabet_size = 0
        self.char_alphabet_size = 0
        self.label_alphabet_size = 0
        self.feature_alphabet_sizes = []
        self.feature_emb_dims = []
        self.norm_feature_embs = []
        self.word_emb_dim = 50
        self.char_emb_dim = 30

        ### Networks
        self.word_feature_extractor = "LSTM"
        self.use_char = True
        self.char_feature_extractor = "CNN"
        self.use_crf = True
        self.nbest = None

        ## Training
        self.average_batch_loss = False
        self.optimizer = "SGD"
        self.status = "train"

        ### Hyperparameters
        self.HP_cnn_layer = 4
        self.HP_iteration = 100
        self.HP_batch_size = 10
        self.HP_char_hidden_dim = 50
        self.HP_hidden_dim = 200
        self.HP_dropout = 0.5
        self.HP_lstm_layer = 1
        self.HP_bilstm = True

        self.HP_gpu = False
        self.HP_lr = 0.015
        self.HP_lr_decay = 0.05
        self.HP_clip = None
        self.HP_momentum = 0
        self.HP_l2 = 1e-8

    def show_data_summary(self):
        print("++" * 50)
        print("DATA SUMMARY START:")
        print(" I/O:")
        if self.sentence_classification:
            print("     Start Sentence Classification task...")
        else:
            print("     Start   Sequence   Laebling   task (Two-Task: LID + Fusion)...")
        print("     Tag          scheme: %s" % (self.tagScheme))
        print("     Split         token: %s" % (self.split_token))
        print("     MAX SENTENCE LENGTH: %s" % (self.MAX_SENTENCE_LENGTH))
        print("     MAX   WORD   LENGTH: %s" % (self.MAX_WORD_LENGTH))
        print("     Number   normalized: %s" % (self.number_normalized))
        print("     Word  alphabet size: %s" % (self.word_alphabet_size))
        print("     Char  alphabet size: %s" % (self.char_alphabet_size))
        print("     Label alphabet size (LID): %s" % (self.label_alphabet_size))
        print("     Fusion label size: %s" % (self.fusion_label_alphabet_size))
        print("     Word embedding  dir: %s" % (self.word_emb_dir))
        print("     Char embedding  dir: %s" % (self.char_emb_dir))
        print("     Word embedding size: %s" % (self.word_emb_dim))
        print("     Char embedding size: %s" % (self.char_emb_dim))
        print("     Norm   word     emb: %s" % (self.norm_word_emb))
        print("     Norm   char     emb: %s" % (self.norm_char_emb))
        print("     Train  file directory: %s" % (self.train_dir))
        print("     Dev    file directory: %s" % (self.dev_dir))
        print("     Test   file directory: %s" % (self.test_dir))
        print("     Raw    file directory: %s" % (self.raw_dir))
        print("     Dset   file directory: %s" % (self.dset_dir))
        print("     Model  file directory: %s" % (self.model_dir))
        print("     Loadmodel   directory: %s" % (self.load_model_dir))
        print("     Decode file directory: %s" % (self.decode_dir))
        print("     Train instance number: %s" % (len(self.train_texts)))
        print("     Dev   instance number: %s" % (len(self.dev_texts)))
        print("     Test  instance number: %s" % (len(self.test_texts)))
        print("     Raw   instance number: %s" % (len(self.raw_texts)))
        print("     FEATURE num: %s" % (self.feature_num))
        for idx in range(self.feature_num):
            print("         Fe: %s  alphabet  size: %s" % (self.feature_alphabets[idx].name, self.feature_alphabet_sizes[idx]))
            print("         Fe: %s  embedding  dir: %s" % (self.feature_alphabets[idx].name, self.feature_emb_dirs[idx]))
            print("         Fe: %s  embedding size: %s" % (self.feature_alphabets[idx].name, self.feature_emb_dims[idx]))
            print("         Fe: %s  norm       emb: %s" % (self.feature_alphabets[idx].name, self.norm_feature_embs[idx]))
        print(" " + "++" * 20)
        print(" Model Network:")
        print("     Model        use_crf: %s" % (self.use_crf))
        print("     Model word extractor: %s" % (self.word_feature_extractor))
        print("     Model       use_char: %s" % (self.use_char))
        if self.use_char:
            print("     Model char extractor: %s" % (self.char_feature_extractor))
            print("     Model char_hidden_dim: %s" % (self.HP_char_hidden_dim))
        print(" " + "++" * 20)
        print(" Training:")
        print("     Optimizer: %s" % (self.optimizer))
        print("     Iteration: %s" % (self.HP_iteration))
        print("     BatchSize: %s" % (self.HP_batch_size))
        print("     Average  batch   loss: %s" % (self.average_batch_loss))
        print(" " + "++" * 20)
        print(" Hyperparameters:")
        print("     Hyper              lr: %s" % (self.HP_lr))
        print("     Hyper        lr_decay: %s" % (self.HP_lr_decay))
        print("     Hyper         HP_clip: %s" % (self.HP_clip))
        print("     Hyper        momentum: %s" % (self.HP_momentum))
        print("     Hyper              l2: %s" % (self.HP_l2))
        print("     Hyper      hidden_dim: %s" % (self.HP_hidden_dim))
        print("     Hyper         dropout: %s" % (self.HP_dropout))
        print("     Hyper      lstm_layer: %s" % (self.HP_lstm_layer))
        print("     Hyper          bilstm: %s" % (self.HP_bilstm))
        print("     Hyper             GPU: %s" % (self.HP_gpu))
        print(" Two-task params:")
        print("     lambda_fusion: %s" % (self.lambda_fusion))
        print("     fusion_pos_weight: %s" % (self.fusion_pos_weight))
        print("DATA SUMMARY END.")
        print("++" * 50)
        sys.stdout.flush()


    def initial_feature_alphabets(self):
        if self.sentence_classification:
            items = open(self.train_dir, 'r').readline().strip('\n').split('\t')
            total_column = len(items)
            # sentence classification: last column is label
            end_col = total_column - 1
        else:
            # sequence labeling: last TWO columns are labels (LID, FUS)
            items = open(self.train_dir, 'r', encoding="utf8").readline().strip('\n').split()
            total_column = len(items)
            end_col = total_column - 2

        if total_column > 3:
            for idx in range(1, end_col):
                feature_prefix = items[idx].split(']', 1)[0] + "]"
                self.feature_alphabets.append(Alphabet(feature_prefix))
                self.feature_name.append(feature_prefix)
                print("Find feature: ", feature_prefix)

        self.feature_num = len(self.feature_alphabets)
        self.pretrain_feature_embeddings = [None] * self.feature_num
        self.feature_emb_dims = [20] * self.feature_num
        self.feature_emb_dirs = [None] * self.feature_num
        self.norm_feature_embs = [False] * self.feature_num
        self.feature_alphabet_sizes = [0] * self.feature_num

        if self.feat_config:
            for idx in range(self.feature_num):
                if self.feature_name[idx] in self.feat_config:
                    self.feature_emb_dims[idx] = self.feat_config[self.feature_name[idx]]['emb_size']
                    self.feature_emb_dirs[idx] = self.feat_config[self.feature_name[idx]]['emb_dir']
                    self.norm_feature_embs[idx] = self.feat_config[self.feature_name[idx]]['emb_norm']


    def build_alphabet(self, input_file):
        in_lines = open(input_file, 'r', encoding="utf8").readlines()
        for line in in_lines:
            if len(line) > 2:
                if self.sentence_classification:
                    pairs = line.strip().split(self.split_token)
                    sent = pairs[0]
                    if sys.version_info[0] < 3:
                        sent = sent.decode('utf-8')
                    words = sent.split()
                    for word in words:
                        if self.number_normalized:
                            word = normalize_word(word)
                        self.word_alphabet.add(word)
                        for ch in word:
                            self.char_alphabet.add(ch)
                    label = pairs[-1]
                    self.label_alphabet.add(label)
                    for idx in range(self.feature_num):
                        feat_idx = pairs[idx + 1].split(']', 1)[-1]
                        self.feature_alphabets[idx].add(feat_idx)
                else:
                    pairs = line.strip().split()
                    word = pairs[0]
                    if sys.version_info[0] < 3:
                        word = word.decode('utf-8')
                    if self.number_normalized:
                        word = normalize_word(word)

                    # last two cols
                    lid_label = pairs[-2]
                    fus_label = pairs[-1]

                    self.label_alphabet.add(lid_label)
                    self.fusion_label_alphabet.add(fus_label)
                    self.word_alphabet.add(word)

                    for idx in range(self.feature_num):
                        feat_idx = pairs[idx + 1].split(']', 1)[-1]
                        self.feature_alphabets[idx].add(feat_idx)
                    for ch in word:
                        self.char_alphabet.add(ch)

        self.word_alphabet_size = self.word_alphabet.size()
        self.char_alphabet_size = self.char_alphabet.size()
        self.label_alphabet_size = self.label_alphabet.size()
        self.fusion_label_alphabet_size = self.fusion_label_alphabet.size()
        for idx in range(self.feature_num):
            self.feature_alphabet_sizes[idx] = self.feature_alphabets[idx].size()

        # tag scheme from LID alphabet only
        startS, startB = False, False
        for label, _ in self.label_alphabet.iteritems():
            if "S-" in label.upper():
                startS = True
            elif "B-" in label.upper():
                startB = True
        if startB:
            self.tagScheme = "BMES" if startS else "BIO"
        if self.sentence_classification:
            self.tagScheme = "Not sequence labeling task"


    def fix_alphabet(self):
        self.word_alphabet.close()
        self.char_alphabet.close()
        self.label_alphabet.close()
        self.fusion_label_alphabet.close()
        for idx in range(self.feature_num):
            self.feature_alphabets[idx].close()


    def build_pretrain_emb(self):
        if self.word_emb_dir:
            print("Load pretrained word embedding, norm: %s, dir: %s" % (self.norm_word_emb, self.word_emb_dir))
            self.pretrain_word_embedding, self.word_emb_dim = build_pretrain_embedding(
                self.word_emb_dir, self.word_alphabet, self.word_emb_dim, self.norm_word_emb)

        if self.char_emb_dir:
            print("Load pretrained char embedding, norm: %s, dir: %s" % (self.norm_char_emb, self.char_emb_dir))
            self.pretrain_char_embedding, self.char_emb_dim = build_pretrain_embedding(
                self.char_emb_dir, self.char_alphabet, self.char_emb_dim, self.norm_char_emb)

        for idx in range(self.feature_num):
            if self.feature_emb_dirs[idx]:
                print("Load pretrained feature %s embedding:, norm: %s, dir: %s" %
                      (self.feature_name[idx], self.norm_feature_embs[idx], self.feature_emb_dirs[idx]))
                self.pretrain_feature_embeddings[idx], self.feature_emb_dims[idx] = build_pretrain_embedding(
                    self.feature_emb_dirs[idx], self.feature_alphabets[idx], self.feature_emb_dims[idx], self.norm_feature_embs[idx])


    def generate_instance(self, name):
        self.fix_alphabet()
        if name == "train":
            self.train_texts, self.train_Ids = read_instance(
                self.train_dir, self.word_alphabet, self.char_alphabet, self.feature_alphabets,
                self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH,
                self.sentence_classification, self.split_token, fusion_label_alphabet=self.fusion_label_alphabet)
        elif name == "dev":
            self.dev_texts, self.dev_Ids = read_instance(
                self.dev_dir, self.word_alphabet, self.char_alphabet, self.feature_alphabets,
                self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH,
                self.sentence_classification, self.split_token, fusion_label_alphabet=self.fusion_label_alphabet)
        elif name == "test":
            self.test_texts, self.test_Ids = read_instance(
                self.test_dir, self.word_alphabet, self.char_alphabet, self.feature_alphabets,
                self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH,
                self.sentence_classification, self.split_token, fusion_label_alphabet=self.fusion_label_alphabet)
        elif name == "raw":
            self.raw_texts, self.raw_Ids = read_instance(
                self.raw_dir, self.word_alphabet, self.char_alphabet, self.feature_alphabets,
                self.label_alphabet, self.number_normalized, self.MAX_SENTENCE_LENGTH,
                self.sentence_classification, self.split_token, fusion_label_alphabet=self.fusion_label_alphabet)
        else:
            print("Error: you can only generate train/dev/test instance! Illegal input:%s" % (name))


    def load(self, data_file):
        f = open(data_file, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)

    def save(self, save_file):
        f = open(save_file, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()


    def read_config(self, config_file):
        config = config_file_to_dict(config_file)

        # read data
        for the_item in ['train_dir', 'dev_dir', 'test_dir', 'raw_dir', 'decode_dir', 'dset_dir', 'model_dir', 'load_model_dir',
                         'word_emb_dir', 'char_emb_dir']:
            if the_item in config:
                setattr(self, the_item, config[the_item])

        for the_item in ['MAX_SENTENCE_LENGTH', 'MAX_WORD_LENGTH', 'word_emb_dim', 'char_emb_dim']:
            if the_item in config:
                setattr(self, the_item, int(config[the_item]))

        for the_item in ['norm_word_emb', 'norm_char_emb', 'number_normalized', 'sentence_classification', 'seg']:
            if the_item in config:
                setattr(self, the_item, str2bool(config[the_item]))

        # NEW two-task params
        if 'lambda_fusion' in config:
            self.lambda_fusion = float(config['lambda_fusion'])
        if 'fusion_pos_weight' in config:
            self.fusion_pos_weight = float(config['fusion_pos_weight'])

        # read network
        if 'use_crf' in config:
            self.use_crf = str2bool(config['use_crf'])
        if 'use_char' in config:
            self.use_char = str2bool(config['use_char'])
        if 'word_seq_feature' in config:
            self.word_feature_extractor = config['word_seq_feature']
        if 'char_seq_feature' in config:
            self.char_feature_extractor = config['char_seq_feature']
        if 'nbest' in config:
            self.nbest = int(config['nbest'])
        if 'feature' in config:
            self.feat_config = config['feature']

        # training setting
        if 'optimizer' in config:
            self.optimizer = config['optimizer']
        if 'ave_batch_loss' in config:
            self.average_batch_loss = str2bool(config['ave_batch_loss'])
        if 'status' in config:
            self.status = config['status']

        # hyperparams
        for the_item, caster in [
            ('cnn_layer', int),
            ('iteration', int),
            ('batch_size', int),
            ('char_hidden_dim', int),
            ('hidden_dim', int),
            ('dropout', float),
            ('lstm_layer', int),
            ('bilstm', str2bool),
            ('gpu', str2bool),
            ('learning_rate', float),
            ('lr_decay', float),
            ('clip', float),
            ('momentum', float),
            ('l2', float),
        ]:
            if the_item in config:
                val = caster(config[the_item])
                if the_item == 'cnn_layer': self.HP_cnn_layer = val
                elif the_item == 'iteration': self.HP_iteration = val
                elif the_item == 'batch_size': self.HP_batch_size = val
                elif the_item == 'char_hidden_dim': self.HP_char_hidden_dim = val
                elif the_item == 'hidden_dim': self.HP_hidden_dim = val
                elif the_item == 'dropout': self.HP_dropout = val
                elif the_item == 'lstm_layer': self.HP_lstm_layer = val
                elif the_item == 'bilstm': self.HP_bilstm = val
                elif the_item == 'gpu': self.HP_gpu = val
                elif the_item == 'learning_rate': self.HP_lr = val
                elif the_item == 'lr_decay': self.HP_lr_decay = val
                elif the_item == 'clip': self.HP_clip = val
                elif the_item == 'momentum': self.HP_momentum = val
                elif the_item == 'l2': self.HP_l2 = val

        if self.sentence_classification:
            self.seg = False
            self.use_crf = False


def config_file_to_dict(input_file):
    config = {}
    fins = open(input_file, 'r').readlines()
    for line in fins:
        if len(line) > 0 and line[0] == "#":
            continue
        if "=" in line:
            pair = line.strip().split('#', 1)[0].split('=', 1)
            item = pair[0]
            if item == "feature":
                if item not in config:
                    feat_dict = {}
                    config[item] = feat_dict
                feat_dict = config[item]
                new_pair = pair[-1].split()
                feat_name = new_pair[0]
                one_dict = {"emb_dir": None, "emb_size": 10, "emb_norm": False}
                if len(new_pair) > 1:
                    for idx in range(1, len(new_pair)):
                        conf_pair = new_pair[idx].split('=')
                        if conf_pair[0] == "emb_dir":
                            one_dict["emb_dir"] = conf_pair[-1]
                        elif conf_pair[0] == "emb_size":
                            one_dict["emb_size"] = int(conf_pair[-1])
                        elif conf_pair[0] == "emb_norm":
                            one_dict["emb_norm"] = str2bool(conf_pair[-1])
                feat_dict[feat_name] = one_dict
            else:
                if item in config:
                    print("Warning: duplicated config item found: %s, updated." % (pair[0]))
                config[item] = pair[-1]
    return config


def str2bool(string):
    return True if string in ["True", "true", "TRUE"] else False
