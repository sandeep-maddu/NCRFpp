from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from .wordsequence import WordSequence
from .crf import CRF

class SeqLabel(nn.Module):
    def __init__(self, data):
        super(SeqLabel, self).__init__()
        self.use_crf = data.use_crf
        print("build sequence labeling network (Two-Task: LID + Fusion)...")
        print("use_char: ", data.use_char)
        if data.use_char:
            print("char feature extractor: ", data.char_feature_extractor)
        print("word feature extractor: ", data.word_feature_extractor)
        print("use crf: ", self.use_crf)

        self.gpu = data.HP_gpu
        self.average_batch = data.average_batch_loss

        # LID
        lid_label_size = data.label_alphabet_size
        data.label_alphabet_size += 2  # NCRF++ internal hack (keep)
        self.word_hidden = WordSequence(data)
        if self.use_crf:
            self.crf = CRF(lid_label_size, self.gpu)

        # Fusion head (auxiliary)
        self.fusion_label_size = data.fusion_label_alphabet_size
        self.hidden2fusion = nn.Linear(data.HP_hidden_dim, self.fusion_label_size)

        # Loss weights
        self.lambda_fusion = float(getattr(data, "lambda_fusion", 0.2))
        self.fusion_pos_weight = float(getattr(data, "fusion_pos_weight", 10.0))

        # Build weight vector for fusion CE:
        # assume labels are ["NF","FUS"] in any order; we weight "FUS" higher if present.
        weight = torch.ones(self.fusion_label_size)
        for lab, idx in data.fusion_label_alphabet.iteritems():
            u = lab.upper().strip()
            # treat any FU-* tag as positive (B-FU / I-FU / S-FU etc.)
            if u.endswith("FU") or u.endswith("-FU") or "FU" in u:
                # but avoid accidental match of NF
                if "NF" not in u:
                    weight[idx] = self.fusion_pos_weight
        self.register_buffer("fusion_class_weight", weight)


        if self.gpu:
            self.hidden2fusion = self.hidden2fusion.cuda()

    def calculate_loss(self, word_inputs, feature_inputs, word_seq_lengths,
                       char_inputs, char_seq_lengths, char_seq_recover,
                       batch_label_lid, batch_label_fus, mask):
        # Get LID emissions + shared encoder features
        lid_emissions, feature_out = self.word_hidden(
            word_inputs, feature_inputs, word_seq_lengths,
            char_inputs, char_seq_lengths, char_seq_recover,
            return_feature=True
        )

        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)

        # --- LID loss (CRF) ---
        if self.use_crf:
            lid_loss = self.crf.neg_log_likelihood_loss(lid_emissions, mask, batch_label_lid)
            _, lid_tag_seq = self.crf._viterbi_decode(lid_emissions, mask)
        else:
            loss_function = nn.NLLLoss(ignore_index=0, reduction='sum')
            outs = lid_emissions.view(batch_size * seq_len, -1)
            score = F.log_softmax(outs, 1)
            lid_loss = loss_function(score, batch_label_lid.view(batch_size * seq_len))
            _, lid_tag_seq = torch.max(score, 1)
            lid_tag_seq = lid_tag_seq.view(batch_size, seq_len)

        # --- Fusion loss (weighted CE over tokens) ---
        fus_logits = self.hidden2fusion(feature_out)  # (B, T, fusion_label_size)
        fus_logits_flat = fus_logits.view(batch_size * seq_len, -1)
        fus_gold_flat = batch_label_fus.view(batch_size * seq_len)

        # mask padded
        mask_flat = mask.view(batch_size * seq_len).bool()
        fus_logits_flat = fus_logits_flat[mask_flat]
        fus_gold_flat = fus_gold_flat[mask_flat]

        fus_loss = F.cross_entropy(
            fus_logits_flat,
            fus_gold_flat,
            weight=self.fusion_class_weight.to(fus_logits_flat.device),
            reduction='sum'
        )

        total_loss = lid_loss + self.lambda_fusion * fus_loss

        if self.average_batch:
            total_loss = total_loss / batch_size

        return total_loss, lid_tag_seq

    def forward(self, word_inputs, feature_inputs, word_seq_lengths,
                char_inputs, char_seq_lengths, char_seq_recover, mask):
        lid_emissions = self.word_hidden(
            word_inputs, feature_inputs, word_seq_lengths,
            char_inputs, char_seq_lengths, char_seq_recover,
            return_feature=False
        )
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)

        if self.use_crf:
            _, tag_seq = self.crf._viterbi_decode(lid_emissions, mask)
        else:
            lid_emissions = lid_emissions.view(batch_size * seq_len, -1)
            _, tag_seq = torch.max(lid_emissions, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)
            tag_seq = mask.long() * tag_seq
        return tag_seq

    def decode_nbest(self, word_inputs, feature_inputs, word_seq_lengths,
                 char_inputs, char_seq_lengths, char_seq_recover, mask, nbest):
        if not self.use_crf:
            print("Nbest output is currently supported only for CRF! Exit...")
            exit(0)
    
        # Get both LID emissions and shared features
        lid_emissions, feature_out = self.word_hidden(
            word_inputs, feature_inputs, word_seq_lengths,
            char_inputs, char_seq_lengths, char_seq_recover,
            return_feature=True
        )
    
        # LID n-best via CRF
        lid_scores, lid_tag_seq = self.crf._viterbi_decode_nbest(lid_emissions, mask, nbest)
    
        # Fusion 1-best via linear head
        fusion_logits = self.hidden2fusion(feature_out)      # [B, T, fusion_label_size]
        fusion_pred = torch.argmax(fusion_logits, dim=-1)    # [B, T]
    
        return lid_scores, lid_tag_seq, fusion_pred
    
