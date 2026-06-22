import os
import argparse
import torch
from tqdm import tqdm

import sys
# replace the path with https://github.com/SuperCarryDFY/b097bade29ef6c984ea1c6f28ec5ece165c3765a459f16085d618a97f9a60286/pretrained_model/ProTrek/sequence_text_protrek_score.py
sys.path.append("path_to_sequence_text_protrek_score")
from model.claprot.claprot_trimodal_model import CLAProtTrimodalModel


def read_fasta(file_path, max_seqs=1000000):
    sequences = {}
    current_header = None
    current_sequence = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_header is not None:
                    # 保存上一个序列
                    sequences[current_header] = "".join(current_sequence)
                    current_sequence = []
                current_header = line[1:]  # 去掉'>'符号
            else:
                current_sequence.append(line)
            if len(sequences) > max_seqs:
                break
        # 保存最后一个序列
        if current_header is not None:
            sequences[current_header] = "".join(current_sequence)

    return sequences


def evaluate_clip_score(model, seq, pred_text, seq_type='prot'):
    if not isinstance(seq, list):
        seq = [seq]
    if not isinstance(pred_text, list):
        pred_text = [pred_text]

    with torch.no_grad():
        if seq_type == 'prot':
            seq_repr = model.get_protein_repr(seq)
        else:
            seq_repr = model.get_structure_repr(seq)
        pred_text_repr = model.get_text_repr(pred_text)

        sim_pred_text = torch.matmul(pred_text_repr, seq_repr.T) / model.temperature

        sim_pred_mask = torch.eye(sim_pred_text.size(0), device=sim_pred_text.device)
        sim_pred_text = sim_pred_text.masked_fill(sim_pred_mask == 0, -1e9)
        sim_pred_text = sim_pred_text.max(dim=1)[0]
        return sim_pred_text

def load_CLIP():
    # loading model
    model_config = {
        "protein_config": "https://github.com/SuperCarryDFY/b097bade29ef6c984ea1c6f28ec5ece165c3765a459f16085d618a97f9a60286/pretrained_model/ProTrek/esm2_t33_650M_UR50D",
        "text_config": "https://github.com/SuperCarryDFY/b097bade29ef6c984ea1c6f28ec5ece165c3765a459f16085d618a97f9a60286/pretrained_model/ProTrek/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "structure_config": "https://github.com/SuperCarryDFY/b097bade29ef6c984ea1c6f28ec5ece165c3765a459f16085d618a97f9a60286/pretrained_model/ProTrek/foldseek_t30_150M",
        "from_checkpoint": "https://github.com/SuperCarryDFY/b097bade29ef6c984ea1c6f28ec5ece165c3765a459f16085d618a97f9a60286/pretrained_model/ProTrek/ProTrek_650M_UniRef50.pt",
        "load_protein_pretrained": False,
        "load_text_pretrained": False,
    }
    model = CLAProtTrimodalModel(**model_config)
    model.to("cuda").eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('res_dir')
    parser.add_argument('--seq_file_name', default="all_seq_token.fasta")
    args = parser.parse_args()

    sequences = []
    name_list = []
    assert os.path.exists(os.path.join(args.res_dir, args.seq_file_name)), f"file {os.path.join(args.res_dir, args.seq_file_name)} not found. "
    if args.seq_file_name.endswith(".fasta"):
        seq_dict = read_fasta(os.path.join(args.res_dir, args.seq_file_name))
        for name in seq_dict.keys():
            name_list.append(name)
            sequences.append(seq_dict[name])
    else:
        with open(os.path.join(args.res_dir, args.seq_file_name), "r") as f:
            for idx, seq in enumerate(f):
                sequences.append(seq.strip())
                name_list.append(f"protrekscore_idx_{idx}")
    
    desc = open(os.path.join(args.res_dir, "desc.txt"), "r").read().strip()

    model = load_CLIP()
    scores = []
    for seq in tqdm(sequences):
        score = evaluate_clip_score(model, seq, desc, seq_type='prot')
        scores.append(score.item())
        
    with open(os.path.join(args.res_dir, f"{args.seq_file_name}_ProTrekScore.tsv"), "w") as f:
        for name, score in zip(name_list, scores):
            f.write(f"{name}\t{score}\n")

if __name__ == "__main__":
    main()