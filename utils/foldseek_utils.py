'''
Modified from Jin Su: https://github.com/westlake-repl/SaProt/blob/main/utils/foldseek_util.py
'''
import os
import time
import numpy as np
import sys

from Bio.PDB import PDBParser, MMCIFParser


sys.path.append(".")


# Get structural seqs from pdb file
def get_struc_seq(foldseek,
                  path,
                  chains: list = None,
                  process_id: int = 0,
                  foldseek_verbose: bool = False) -> dict:
    """

    Args:
        foldseek: Binary executable file of foldseek

        path: Path to pdb file

        chains: Chains to be extracted from pdb file. If None, all chains will be extracted.

        process_id: Process ID for temporary files. This is used for parallel processing.

        foldseek_verbose: If True, foldseek will print verbose messages.

    Returns:
        seq_dict: A dict of structural seqs. The keys are chain IDs. The values are tuples of
        (seq, struc_seq, combined_seq).
    """
    assert os.path.exists(foldseek), f"Foldseek not found: {foldseek}"
    assert os.path.exists(path), f"PDB file not found: {path}"
    
    tmp_save_path = f"get_struc_seq_{process_id}_{time.time()}.tsv"
    if foldseek_verbose:
        cmd = f"{foldseek} structureto3didescriptor --threads 1 --chain-name-mode 1 {path} {tmp_save_path}"
    else:
        cmd = f"{foldseek} structureto3didescriptor -v 0 --threads 1 --chain-name-mode 1 {path} {tmp_save_path}"
    os.system(cmd)
    print(cmd)
    seq_dict = {}
    name = os.path.basename(path)
    with open(tmp_save_path, "r") as r:
        for i, line in enumerate(r):
            desc, seq, struc_seq = line.split("\t")[:3]
            
            name_chain = desc.split(" ")[0]
            chain = name_chain.replace(name, "").split("_")[-1]
            
            if chains is None or chain in chains:
                if chain not in seq_dict:
                    combined_seq = "".join([a + b.lower() for a, b in zip(seq, struc_seq)])
                    seq_dict[chain] = (seq, struc_seq, combined_seq)
    
    os.remove(tmp_save_path)
    os.remove(tmp_save_path + ".dbtype")
    return seq_dict
