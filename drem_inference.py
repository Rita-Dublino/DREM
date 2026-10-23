#!/usr/bin/env python3
"""
DREM (Deep Resistance gene Evaluator Model) - inference script.

Scores protein sequences with the trained DREM binary resistance model and
multi-label domain model, and assigns each protein to a prioritization
category.

Usage:
    python3 drem_inference.py -f proteins.fasta -o drem_results.tsv -m models/
"""

import argparse
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}
MAX_LENGTH = 1024
STRIDE = 512
BATCH_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim, attn_dim=128):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, H, mask):
        scores = self.v(torch.tanh(self.proj(H))).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        alpha = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (H * alpha).sum(dim=1)


class ResistanceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(AMINO_ACIDS) + 1, 32, padding_idx=0)
        self.lstm = nn.LSTM(32, 64, num_layers=2, batch_first=True, bidirectional=True)
        self.attn_pool = AttentionPooling(128)
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        H, _ = self.lstm(self.embedding(x))
        return self.fc(self.attn_pool(H, x != 0)).squeeze(-1)


class DomainModel(nn.Module):
    def __init__(self, n_labels=5):
        super().__init__()
        self.embedding = nn.Embedding(len(AMINO_ACIDS) + 1, 32, padding_idx=0)
        self.lstm = nn.LSTM(32, 64, num_layers=2, batch_first=True, bidirectional=True)
        self.attn_pool = AttentionPooling(128)
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_labels)
        )

    def forward(self, x):
        H, _ = self.lstm(self.embedding(x))
        return self.fc(self.attn_pool(H, x != 0))


class FastDataset(Dataset):
    def __init__(self, sequences):
        self.data = [
            torch.tensor(
                [AA_TO_IDX.get(aa, 0) for aa in seq[:MAX_LENGTH]],
                dtype=torch.long
            )
            for seq in sequences
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    return nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)


def read_fasta(path):
    handle = gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")

    ids = []
    seqs = []
    header = None
    current_seq = []

    for line in handle:
        line = line.strip()

        if not line:
            continue

        if line.startswith(">"):
            if header is not None:
                ids.append(header.split()[0])
                seqs.append("".join(current_seq).upper().rstrip("*."))

            header = line[1:]
            current_seq = []
        else:
            current_seq.append(line)

    if header is not None:
        ids.append(header.split()[0])
        seqs.append("".join(current_seq).upper().rstrip("*."))

    handle.close()
    return ids, seqs


def split_sequence_windows(seq, max_length=1024, stride=512):
    seq = seq.upper()

    if len(seq) <= max_length:
        return [seq]

    windows = []

    for start in range(0, len(seq), stride):
        window = seq[start:start + max_length]

        if len(window) > 0:
            windows.append(window)

        if start + max_length >= len(seq):
            break

    return windows


def safe_load_model(model, path):
    try:
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=True)
        except Exception:
            # Checkpoints saved with extra (non-tensor) objects need full loading
            checkpoint = torch.load(path, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        return model

    except Exception as error:
        print(f"[ERROR] Failed to load model {path}: {str(error)}", file=sys.stderr)
        sys.exit(1)


def classify_prediction(r_val, nbs_p, lrr_p, tir_p, cc_p, kin_p):
    """
    Final hierarchical DREM interpretation logic.
    Important principle:
    Strong NBS signal should never be reported as simple Non_resistance.
    """

    is_nlr = (nbs_p > 0.60) and (r_val > 0.60)

    is_rlk = (
        (kin_p > 0.85) and
        (lrr_p > 0.40) and
        (r_val > 0.80)
    )

    is_rlp = (
        (r_val > 0.90) and
        (lrr_p > 0.80) and
        (nbs_p < 0.30) and
        (kin_p < 0.30)
    )

    is_kinase_r = (
        (r_val > 0.90) and
        (kin_p > 0.85) and
        (lrr_p < 0.30) and
        (nbs_p < 0.30)
    )

    max_acc = max(lrr_p, tir_p, cc_p)

    nlr_scs = nbs_p * max_acc
    rlk_scs = kin_p * lrr_p
    rlp_scs = (r_val * lrr_p) if is_rlp else 0
    kinase_scs = (r_val * kin_p) if is_kinase_r else 0

    scs_val = max(nlr_scs, rlk_scs, rlp_scs, kinase_scs)
    prs_val = np.sqrt((r_val**2 + scs_val**2) / 2)

    # A. Canonical profiles

    if is_rlk:
        label = "High_Confidence" if prs_val > 0.75 else "Candidate"
        return scs_val, prs_val, label, "RLK", "Canonical_RLK_signal"

    if is_rlp:
        label = "High_Confidence" if prs_val > 0.75 else "Candidate"
        return scs_val, prs_val, label, "RLP", "Canonical_RLP_signal"

    if is_kinase_r:
        label = "High_Confidence" if prs_val > 0.75 else "Candidate"
        return scs_val, prs_val, label, "Kinase_resistance", "Kinase_only_resistance_signal"

    if is_nlr:
        label = "High_Confidence" if prs_val > 0.75 else "Candidate"

        if tir_p > 0.45:
            return scs_val, prs_val, label, "TNL", "Canonical_TNL_signal"
        elif cc_p > 0.40:
            return scs_val, prs_val, label, "CNL", "Canonical_CNL_signal"
        else:
            return scs_val, prs_val, label, "NLR", "Canonical_NLR_signal"

    # B. Strong NBS-supported resistance-like profiles, even if R_Global is low

    if nbs_p > 0.70:
        if tir_p > 0.45:
            return (
                scs_val,
                prs_val,
                "Domain_supported_resistance_like",
                "TNL_like",
                "Strong_NBS_TIR_signal_but_low_global_R"
            )

        elif cc_p > 0.05:
            return (
                scs_val,
                prs_val,
                "Domain_supported_resistance_like",
                "RNL_like",
                "Strong_NBS_CC_signal_but_low_global_R"
            )

        else:
            return (
                scs_val,
                prs_val,
                "Domain_supported_resistance_like",
                "NLR_like",
                "Strong_NBS_signal_but_low_global_R"
            )

    # C. Strong global R without canonical domain architecture

    if (
        (r_val > 0.90) and
        (nbs_p < 0.40) and
        (kin_p < 0.40) and
        (lrr_p < 0.70)
    ):
        return (
            scs_val,
            prs_val,
            "Low_Confidence/Divergent",
            "Non_canonical",
            "Strong_global_R_without_canonical_domains"
        )

    # D. Other strong domain signals but low/insufficient global support

    if (
        (r_val < 0.60) and
        (
            (kin_p > 0.85) or
            (lrr_p > 0.85) or
            (tir_p > 0.85) or
            (cc_p > 0.85)
        )
    ):
        if kin_p > 0.85:
            return (
                scs_val,
                prs_val,
                "Domain_supported_resistance_like",
                "Kinase_like",
                "Strong_Kinase_signal_but_low_global_R"
            )

        if lrr_p > 0.85:
            return (
                scs_val,
                prs_val,
                "Domain_supported_resistance_like",
                "LRR_like",
                "Strong_LRR_signal_but_low_global_R"
            )

        if tir_p > 0.85:
            return (
                scs_val,
                prs_val,
                "Domain_supported_resistance_like",
                "TIR_like",
                "Strong_TIR_signal_but_low_global_R"
            )

        if cc_p > 0.85:
            return (
                scs_val,
                prs_val,
                "Domain_supported_resistance_like",
                "CC_like",
                "Strong_CC_signal_but_low_global_R"
            )

    # E. Intermediate/divergent signals

    if (
        (r_val > 0.40) and
        (
            (nbs_p > 0.40) or
            (kin_p > 0.40) or
            (lrr_p > 0.70) or
            (tir_p > 0.45) or
            (cc_p > 0.40)
        )
    ):
        if nbs_p > 0.40:
            cls = "NLR_like"
        elif kin_p > 0.40:
            cls = "Kinase_like"
        elif lrr_p > 0.70:
            cls = "LRR_like"
        elif tir_p > 0.45:
            cls = "TIR_like"
        elif cc_p > 0.40:
            cls = "CC_like"
        else:
            cls = "Divergent"

        return (
            scs_val,
            prs_val,
            "Low_Confidence/Divergent",
            cls,
            "Weak_or_divergent_resistance_signal"
        )

    # F. No detectable canonical resistance signal

    return (
        scs_val,
        prs_val,
        "Non_resistance_like",
        "None",
        "No_detectable_resistance_signal"
    )


def run_inference(fasta_path, model_dir, output_file):
    print(f"[DREM] Loading FASTA: {fasta_path}", file=sys.stderr)
    ids, sequences = read_fasta(fasta_path)
    print(f"[DREM] Loaded {len(ids)} original sequences", file=sys.stderr)

    if len(sequences) == 0:
        print("[ERROR] No sequences found in FASTA file", file=sys.stderr)
        sys.exit(1)

    window_ids = []
    window_parent_ids = []
    window_sequences = []
    window_starts = []
    window_ends = []

    for protein_id, sequence in zip(ids, sequences):
        windows = split_sequence_windows(sequence, max_length=MAX_LENGTH, stride=STRIDE)

        for w_idx, window in enumerate(windows):
            start = (w_idx * STRIDE) + 1
            end = start + len(window) - 1

            window_parent_ids.append(protein_id)
            window_ids.append(f"{protein_id}_window{w_idx + 1}_{start}_{end}")
            window_sequences.append(window)
            window_starts.append(start)
            window_ends.append(end)

    print(f"[DREM] Created {len(window_sequences)} sequence windows", file=sys.stderr)

    dataset = FastDataset(window_sequences)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    model_dir = Path(model_dir)
    res_model_path = model_dir / "resistance_model.pt"
    dom_model_path = model_dir / "domain_model.pt"

    if not res_model_path.exists():
        print(f"[ERROR] Resistance model not found: {res_model_path}", file=sys.stderr)
        sys.exit(1)

    if not dom_model_path.exists():
        print(f"[ERROR] Domain model not found: {dom_model_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[DREM] Loading models from {model_dir}", file=sys.stderr)

    res_model = safe_load_model(ResistanceModel().to(device), str(res_model_path))
    dom_model = safe_load_model(DomainModel().to(device), str(dom_model_path))

    print("[DREM] Running inference on windows...", file=sys.stderr)

    all_R = []
    all_D = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Processing"):
            batch = batch.to(device)
            all_R.append(torch.sigmoid(res_model(batch)).cpu().numpy())
            all_D.append(torch.sigmoid(dom_model(batch)).cpu().numpy())

    R_window = np.concatenate(all_R)
    D_window = np.vstack(all_D)

    df_windows = pd.DataFrame({
        "Protein_ID": window_parent_ids,
        "Window_ID": window_ids,
        "Window_Start": window_starts,
        "Window_End": window_ends,
        "R_Global": R_window,
        "P_NBS": D_window[:, 0],
        "P_LRR": D_window[:, 1],
        "P_TIR": D_window[:, 2],
        "P_CC": D_window[:, 3],
        "P_Kinase": D_window[:, 4]
    })

    df_agg = df_windows.groupby("Protein_ID", as_index=False).agg({
        "R_Global": "max",
        "P_NBS": "max",
        "P_LRR": "max",
        "P_TIR": "max",
        "P_CC": "max",
        "P_Kinase": "max"
    })

    print("[DREM] Calculating final classification...", file=sys.stderr)

    final_scs = []
    final_prs = []
    labels = []
    classes = []
    evidence_profiles = []

    for _, row in df_agg.iterrows():
        scs, prs, label, cls, evidence = classify_prediction(
            row["R_Global"],
            row["P_NBS"],
            row["P_LRR"],
            row["P_TIR"],
            row["P_CC"],
            row["P_Kinase"]
        )

        final_scs.append(scs)
        final_prs.append(prs)
        labels.append(label)
        classes.append(cls)
        evidence_profiles.append(evidence)

    df_agg["SCS"] = final_scs
    df_agg["PRS_Final"] = final_prs
    df_agg["Label"] = labels
    df_agg["Class"] = classes
    df_agg["Evidence_Profile"] = evidence_profiles

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".xlsx":
        df_agg.to_excel(output_path, index=False)
    else:
        df_agg.to_csv(output_path, sep="\t", index=False)

    window_output = str(output_path.with_name(output_path.stem + "_windows.tsv"))
    df_windows.to_csv(window_output, sep="\t", index=False)

    print(f"[DREM] Saving final results to {output_file}", file=sys.stderr)
    print(f"[DREM] Saving window-level results to {window_output}", file=sys.stderr)

    print("[DREM] ============== Summary ==============", file=sys.stderr)
    print(f"[DREM] Total proteins: {len(df_agg)}", file=sys.stderr)
    print(f"[DREM] Total windows: {len(df_windows)}", file=sys.stderr)
    print(df_agg["Label"].value_counts().to_string(), file=sys.stderr)
    print("[DREM] ========================================", file=sys.stderr)
    print("[DREM] Pipeline complete!", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DREM inference")

    parser.add_argument("-f", "--fasta", required=True, help="Input protein FASTA file (.fasta or .fasta.gz)")
    parser.add_argument("-m", "--models", default="./models", help="Model directory (default: ./models)")
    parser.add_argument("-o", "--output", default="drem_results.tsv", help="Output file, .tsv or .xlsx (default: drem_results.tsv)")

    args = parser.parse_args()

    if not Path(args.fasta).exists():
        print(f"[ERROR] FASTA file not found: {args.fasta}", file=sys.stderr)
        sys.exit(1)

    run_inference(args.fasta, args.models, args.output)
