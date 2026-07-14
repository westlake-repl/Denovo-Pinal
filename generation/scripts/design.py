import torch 
import os
import logging
from rich.logging import RichHandler
import argparse
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import set_seed
from datetime import timedelta
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils.design_utils import load_pinal, PinalDesign

FORMAT = "%(message)s"
logging.basicConfig(level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--desc_path", type=str, help="description file path.", required=True)
    parser.add_argument("--num", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.)
    parser.add_argument("--saprot_sample_type", choices=["argmax", "multinomial"], default="argmax")
    parser.add_argument("--multinomial_temperature", type=float, default=0.3)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    assert not (args.saprot_sample_type == "argmax" and args.multinomial_temperature != 0.3) , "If you choose argmax, the specific multinomial_temperature would be of no use."

    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=4))
    accelerator = Accelerator(kwargs_handlers=[timeout_kwargs])
    with open(args.desc_path) as f:
        desc_list = [desc.strip() for desc in f if desc.strip()]
    if not desc_list:
        raise ValueError(f"No prompts found in {args.desc_path}")
    if args.num % len(desc_list) != 0:
        raise ValueError(f"--num ({args.num}) must be divisible by the number of prompts ({len(desc_list)})")
    design_num = args.num // len(desc_list)
    set_seed(args.seed, device_specific=True)
    
    # datetime_str = datetime.now().strftime("%Y%m%d")
    output_dir = os.path.join(args.output_dir,  os.path.basename(args.desc_path).split(".")[0] )

    # configure logger
    rank = accelerator.process_index

    class RankFilter(logging.Filter):
        def filter(self, record):
            record.rank = rank
            return True

    for handler in logging.getLogger().handlers:
        handler.addFilter(RankFilter())
        handler.setFormatter(logging.Formatter("[rank %(rank)s] %(message)s"))

    t2struc, text_tokenizer, structure_tokenizer, saprot, saprot_text_tokenizer, saprot_tokenizer = load_pinal()

    rank = accelerator.process_index
    design_num, remainder = divmod(design_num, accelerator.num_processes)
    design_num += int(rank < remainder)
    
    res = []
    for desc in desc_list:
        print(f"Rank: {rank}, Design number: {design_num}")
        _res = PinalDesign(desc, design_num, t2struc, text_tokenizer, structure_tokenizer, saprot, \
                        saprot_text_tokenizer, saprot_tokenizer, args.temperature, args.saprot_sample_type, args.multinomial_temperature)
        res.extend(_res)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # write desc to file
    if rank == 0:
        with open(os.path.join(output_dir, "desc.txt"), "w") as f:
            f.write("\n".join(desc_list))
        # save seed and args
        with open(os.path.join(output_dir, f"args.txt"), "w") as f:
            f.write(f"Seed: {args.seed}\n")
            f.write(f"Args: {args}\n")
    
    # write sequence to file
    with open(os.path.join(output_dir, f"sequence_rank{rank}.txt"), "w") as f:
        for i, tpl in enumerate(res):
            f.write(f"{tpl[2]}\n")
    
    # write res to tsv file
    with open(os.path.join(output_dir, f"res_rank{rank}.tsv"), "w") as f:
        f.write("Structure\tStructure_logp\tSequence\tSequence_logp\n")
        for i, tpl in enumerate(res):
            f.write(f"{tpl[0]}\t{tpl[1]}\t{tpl[2]}\t{tpl[3]}\n")
    accelerator.wait_for_everyone()
