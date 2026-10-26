# DREM: Deep Resistance gene Evaluator Model

DREM scores plant protein sequences for resistance-associated features and ranks them for downstream analysis. It uses two BiLSTM networks with additive (Bahdanau-style) attention pooling, both working on raw amino acid sequences:

- a **binary resistance model**, which returns an overall resistance score (R-score);
- a **multi-label domain model**, which returns the probability of five resistance-associated domains: NBS, LRR, TIR, CC and Kinase.

The two outputs are combined into a Structural Confidence Score (SCS) and a Protein Resistance Score (PRS). Each protein is then assigned to a prioritization category and to a predicted architecture class.

DREM is integrated in [PRGdb 5.0](https://prgdb.org/prgdb5), where it can be run from the web interface without registration. This repository contains the inference code. The trained model weights are not included and are available from the corresponding author upon request for non-commercial research (see [License](#license)).

## Citation

If you use DREM, please cite:

> Dublino R, Carandente V, Sperlì G, Ercolano MR. DREM: integrating resistance and domain probabilities to prioritize plant immune receptors. *bioRxiv* (2026). https://doi.org/10.64898/2026.10.21.760626

## Installation

Requirements: Python ≥ 3.8 and the packages in `requirements.txt` (PyTorch, NumPy, pandas, tqdm, openpyxl). DREM was developed with Python 3.10 and PyTorch 2.0.1. A GPU is used automatically if available; otherwise DREM runs on CPU.

```bash
git clone https://github.com/Rita-Dublino/DREM.git
cd DREM
pip install -r requirements.txt
```

Place the trained model files (`resistance_model.pt` and `domain_model.pt`) in a folder named `models/` inside the repository.

## Usage

```bash
bash drem.sh -f example/example.fasta -o drem_output
```

or directly with Python:

```bash
python3 drem_inference.py -f example/example.fasta -o drem_output/drem_results.tsv -m models
```

| Option | Description |
|---|---|
| `-f`, `--fasta` | Input protein FASTA file (`.fasta` or `.fasta.gz`), required |
| `-o`, `--output` | `drem.sh`: output directory (default `./drem_output`). `drem_inference.py`: output file, `.tsv` or `.xlsx` (default `drem_results.tsv`) |
| `-m`, `--models` | `drem_inference.py` only: model directory (default `./models`) |

## Input

Protein sequences in FASTA format. The sequence identifier is the first word of the header line. Terminal stop codons (`*`) are removed. Characters other than the 20 standard amino acids are not encoded.

Sequences longer than 1,024 residues are split into 1,024-residue windows with a step of 512 residues. Each window is scored separately, and the protein-level R-score and domain probabilities are the maximum values across its windows.

`example/example.fasta` contains three curated resistance proteins from PRGdb: Roq1 (TNL, *Nicotiana benthamiana*), Pm37 (CNL, *Triticum monococcum*) and ZmWAK-RLK1 (RLK, *Zea mays*).

## Output

`drem_results.tsv` contains one row per protein:

| Column | Description |
|---|---|
| `Protein_ID` | Sequence identifier |
| `R_Global` | R-score, probability from the binary resistance model (0–1) |
| `P_NBS`, `P_LRR`, `P_TIR`, `P_CC`, `P_Kinase` | Domain probabilities from the multi-label model (0–1) |
| `SCS` | Structural Confidence Score |
| `PRS_Final` | Protein Resistance Score |
| `Label` | Prioritization category |
| `Class` | Predicted architecture class |
| `Evidence_Profile` | Short description of the signal behind the assignment |

`drem_results_windows.tsv` reports the same scores for each sequence window (`Window_ID`, `Window_Start`, `Window_End`).

### Prioritization categories

| Label | Meaning |
|---|---|
| `High_Confidence` | Canonical architecture profile with PRS > 0.75 |
| `Candidate` | Canonical architecture profile with PRS ≤ 0.75 |
| `Low_Confidence/Divergent` | Partial or non-canonical evidence, such as a high R-score without canonical domains |
| `Domain_supported_resistance_like` | Strong domain signal with weaker global resistance support |
| `Non_resistance_like` | No sufficient resistance-associated signal |

Canonical profiles are evaluated in this order:

| Class | Conditions |
|---|---|
| `RLK` | P_Kinase > 0.85, P_LRR > 0.40, R > 0.80 |
| `RLP` | R > 0.90, P_LRR > 0.80, P_NBS < 0.30, P_Kinase < 0.30 |
| `Kinase_resistance` | R > 0.90, P_Kinase > 0.85, P_LRR < 0.30, P_NBS < 0.30 |
| `TNL` / `CNL` / `NLR` | P_NBS > 0.60 and R > 0.60; TNL if P_TIR > 0.45, otherwise CNL if P_CC > 0.40, otherwise NLR |

Non-canonical assignments use the suffix `_like` (e.g. `TNL_like`, `NLR_like`, `Kinase_like`, `LRR_like`), `Non_canonical` or `Divergent`. Proteins without a resistance signal have class `None`.

These thresholds are used to rank candidates and do not define absolute biological boundaries. Definitions of the scores and the validation of the models are described in the preprint.

## Repository structure

```
DREM/
├── drem.sh              # command-line wrapper
├── drem_inference.py    # model definition, scoring and classification
├── models/             # trained models, available upon request (not included)
├── example/
│   └── example.fasta
├── requirements.txt
├── CITATION.cff
└── LICENSE              # AGPL-3.0 (code)
```

## License

- **Code** (`drem.sh`, `drem_inference.py`): GNU Affero General Public License v3.0 ([LICENSE](LICENSE)).
- **Trained model weights**: not distributed in this repository. They are available from the corresponding author upon request, for non-commercial research only, under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International licence (CC BY-NC-SA 4.0). Commercial use of the models requires the written permission of the authors.

DREM predictions are freely available through the PRGdb 5.0 web service (https://prgdb.org/prgdb5).

## Contact

For questions or problems, please open an issue in this repository or contact the corresponding author, Maria Raffaella Ercolano (ercolano@unina.it), Department of Agricultural Sciences, University of Naples Federico II, Italy.
