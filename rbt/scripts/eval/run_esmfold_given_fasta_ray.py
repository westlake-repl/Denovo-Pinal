import argparse
import json
import logging
import os

import biotite.structure.io as bsio
import esm
import numpy as np
import ray
import torch
from collections import defaultdict
from ray.util.actor_pool import ActorPool
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [ESMFold] - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_pLDDT(pdb_file):
    struct = bsio.load_structure(pdb_file, extra_fields=["b_factor"])
    return struct.b_factor.mean()


def calculate_pae(out):
    pae = (out["aligned_confidence_probs"][0].cpu().numpy() * np.arange(64)).mean(
        -1
    ) * 31
    mask = out["atom37_atom_exists"][0, :, 1] == 1
    mask = mask.cpu()
    pae = pae[mask, :][:, mask]
    return np.mean(pae)


@ray.remote(num_gpus=1)
class ESMFoldPredictor:
    def __init__(self, cache_dir):
        os.environ["TORCH_HOME"] = cache_dir

        logger.info("Loading ESMFold model on GPU...")
        self.model = esm.pretrained.esmfold_v1().eval().cuda()
        logger.info("Model loaded successfully.")

    def process_batch(self, batch_data, output_path):
        """
        处理一个小批次的数据
        batch_data: list of (entry_id, sequence)
        """
        batch_metrics_dic = defaultdict(dict)

        for entry_id, sequence in batch_data:
            sequence = sequence[:1024]  # 截断
            output_file = f"sequence_{entry_id}.pdb"
            file_full_path = os.path.join(output_path, output_file)

            # try:
            with torch.no_grad():
                out = self.model.infer([sequence])
                pdb_out = self.model.output_to_pdb(out)[0]

            pae_val = calculate_pae(out)
            # batch_pae_dic[output_file] = pae_val
            batch_metrics_dic[output_file]["plddt"] = out["mean_plddt"][0].item()
            batch_metrics_dic[output_file]["ptm"] = out["ptm"][0].item()
            batch_metrics_dic[output_file]["pae"] = pae_val

            with open(file_full_path, "w") as f:
                f.write(pdb_out)


            # except Exception as e:
            #     logger.error(f"Error predicting {entry_id}: {e}")
            #     raise e
                # continue
        return batch_metrics_dic


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch")))
    args = parser.parse_args()

    output_path = args.fasta_path.replace(".fasta", "_esmfold_results")
    os.makedirs(output_path, exist_ok=True)

    ray.init()

    sequences_dict = read_fasta(args.fasta_path)
    logger.info(f"Reading sequences from {args.fasta_path}")

    all_items = []
    for name, seq in sequences_dict.items():
        all_items.append((name, seq))

    logger.info(f"Total sequences: {len(all_items)}")

    num_gpus = torch.cuda.device_count()
    logger.info(f"Detected {num_gpus} GPUs. Spawning actors...")

    actors = [ESMFoldPredictor.remote(args.cache_dir) for _ in range(num_gpus)]
    pool = ActorPool(actors)

    BATCH_SIZE = 5
    chunks = [
        all_items[i : i + BATCH_SIZE] for i in range(0, len(all_items), BATCH_SIZE)
    ]

    logger.info(
        f"Split data into {len(chunks)} batches (size={BATCH_SIZE}). Starting inference..."
    )

    final_batch_metrics_dic = {}

    pbar = tqdm(total=len(chunks), desc="Processing Batches", unit="batch")

    for batch_metrics_dic in pool.map_unordered(
        lambda actor, value: actor.process_batch.remote(value, output_path), chunks
    ):
        final_batch_metrics_dic.update(batch_metrics_dic)
        pbar.update(1)

    pbar.close()

    logger.info("Inference done. Calculating metrics...")
    mean_pae = np.mean([v["pae"] for v in final_batch_metrics_dic.values()])
    mean_plddt = np.mean([v["plddt"] for v in final_batch_metrics_dic.values()])
    mean_ptm = np.mean([v["ptm"] for v in final_batch_metrics_dic.values()])

    logger.info(f"Mean pLDDT: {mean_plddt:.4f}")
    logger.info(f"Mean PAE: {mean_pae:.4f}")
    logger.info(f"Mean pTM: {mean_ptm:.4f}")

    results_path = args.fasta_path.replace(".fasta", "_esmfold_results.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            result_dic = json.load(f)
    else:
        result_dic = {}

    result_dic["ESMFold pLDDT"] = float(mean_plddt)
    result_dic["ESMFold pae"] = float(mean_pae)
    result_dic["ESMFold pTM"] = float(mean_ptm)
    for k, v in final_batch_metrics_dic.items():
        result_dic[f"{k}_metrics"] = v

    with open(results_path, "w") as f:
        json.dump(result_dic, f, indent=4)

    logger.info(f"Metrics saved to {results_path}")


if __name__ == "__main__":
    main()
