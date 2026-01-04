from collections import Counter
import sys

LID_LABELS = ["TE", "EN", "NER", "O"]
FUS_LABELS = ["FU", "O"]

def norm_lid(tag: str) -> str:
    tag = tag.strip()
    if tag == "O":
        return "O"
    tag_u = tag.upper()
    if tag_u.startswith("B-"):
        tag_u = tag_u[2:]
    if tag_u in {"TE", "EN", "NER"}:
        return tag_u
    return tag_u  # keep unexpected labels visible

def norm_fus(tag: str) -> str:
    tag = tag.strip()
    if tag == "NA":
        return "NA"
    tag_u = tag.upper()
    if tag_u.startswith("B-"):
        tag_u = tag_u[2:]
    return "FU" if tag_u == "FU" else "O"

def read_decoded_5col(path: str):
    """
    Reads:
      WORD  GOLD_LID  PRED_LID  GOLD_FUS  PRED_FUS
    Blank line separates sentences.
    """
    gold_lid_sents, pred_lid_sents = [], []
    gold_fus_sents, pred_fus_sents = [], []

    gl, pl, gf, pf = [], [], [], []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if gl:
                    gold_lid_sents.append(gl); pred_lid_sents.append(pl)
                    gold_fus_sents.append(gf); pred_fus_sents.append(pf)
                    gl, pl, gf, pf = [], [], [], []
                continue

            parts = line.split()  # works for tabs/spaces
            if len(parts) < 5:
                continue

            _, gold_lid, pred_lid, gold_fus, pred_fus = parts[0], parts[1], parts[2], parts[3], parts[4]
            gl.append(norm_lid(gold_lid))
            pl.append(norm_lid(pred_lid))
            gf.append(norm_fus(gold_fus))
            pf.append(norm_fus(pred_fus))

    if gl:
        gold_lid_sents.append(gl); pred_lid_sents.append(pl)
        gold_fus_sents.append(gf); pred_fus_sents.append(pf)

    return gold_lid_sents, pred_lid_sents, gold_fus_sents, pred_fus_sents

def token_metrics(gold_sents, pred_sents, labels, zero_division=0, ignore_pred_values=None):
    tp, fp, fn = Counter(), Counter(), Counter()
    correct = 0
    total = 0

    for g_sent, p_sent in zip(gold_sents, pred_sents):
        if len(g_sent) != len(p_sent):
            raise ValueError(f"Sentence length mismatch: gold={len(g_sent)} pred={len(p_sent)}")
        for g, p in zip(g_sent, p_sent):
            if ignore_pred_values and p in ignore_pred_values:
                continue
            total += 1
            if g == p:
                correct += 1
                tp[g] += 1
            else:
                fp[p] += 1
                fn[g] += 1

    acc = correct / total if total else 0.0

    per = {}
    for lab in labels:
        TP, FP, FN = tp[lab], fp[lab], fn[lab]
        prec = TP / (TP + FP) if (TP + FP) else zero_division
        rec  = TP / (TP + FN) if (TP + FN) else zero_division
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) else zero_division
        per[lab] = (prec, rec, f1)

    macro_f1 = sum(per[l][2] for l in labels) / len(labels) if labels else 0.0

    sum_tp = sum(tp[l] for l in labels)
    sum_fp = sum(fp[l] for l in labels)
    sum_fn = sum(fn[l] for l in labels)
    micro_p = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) else zero_division
    micro_r = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) else zero_division
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) else zero_division

    return acc, micro_f1, macro_f1, per, total

def subset_lid(gold_lid_sents, pred_lid_sents, gold_fus_sents, want_fus: str):
    """
    want_fus = "FU"  -> tokens where GOLD_FUS == FU
    want_fus = "O"   -> tokens where GOLD_FUS == O (i.e., non-fusion)
    """
    g2, p2 = [], []
    for gl, pl, gf in zip(gold_lid_sents, pred_lid_sents, gold_fus_sents):
        gl_sub, pl_sub = [], []
        for lid_g, lid_p, fus_g in zip(gl, pl, gf):
            if fus_g == want_fus:
                gl_sub.append(lid_g)
                pl_sub.append(lid_p)
        if gl_sub:
            g2.append(gl_sub)
            p2.append(pl_sub)
    if not g2:
        return None
    return token_metrics(g2, p2, labels=LID_LABELS)

def print_block(title, res):
    acc, micro_f1, macro_f1, per, n = res
    print(f"{title}: N={n}  Acc={acc:.6f}  MicroF1={micro_f1:.6f}  MacroF1={macro_f1:.6f}")
    print(f"{title} per-label F1:", {k: round(v[2], 4) for k, v in per.items()})

def main():
    if len(sys.argv) < 2:
        print("Usage: python eval_lid_fusion.py <raw.decoded.txt>")
        sys.exit(1)

    decoded_path = sys.argv[1]
    gold_lid_sents, pred_lid_sents, gold_fus_sents, pred_fus_sents = read_decoded_5col(decoded_path)

    # 1) LID overall
    overall = token_metrics(gold_lid_sents, pred_lid_sents, labels=LID_LABELS)
    print_block("LID overall", overall)

    # 2) Fusion decoding availability
    fus_has_real_preds = any(p != "NA" for sent in pred_fus_sents for p in sent)
    if not fus_has_real_preds:
        print("Fusion: PRED_FUS is NA everywhere -> fusion decoding is NOT produced/passed (column 5 unusable).")
    else:
        fus_overall = token_metrics(gold_fus_sents, pred_fus_sents, labels=FUS_LABELS, ignore_pred_values={"NA"})
        print_block("Fusion overall", fus_overall)

    # 3) LID on FU tokens only (gold)
    fu_res = subset_lid(gold_lid_sents, pred_lid_sents, gold_fus_sents, want_fus="FU")
    if fu_res is None:
        print("LID on FU tokens: no FU tokens found in GOLD_FUS.")
    else:
        print_block("LID on FU tokens", fu_res)

    # 4) LID on NON-FU tokens (gold O)
    nonfu_res = subset_lid(gold_lid_sents, pred_lid_sents, gold_fus_sents, want_fus="O")
    if nonfu_res is None:
        print("LID on non-FU tokens: no non-FU tokens found in GOLD_FUS (unexpected).")
    else:
        print_block("LID on non-FU tokens", nonfu_res)

if __name__ == "__main__":
    main()
