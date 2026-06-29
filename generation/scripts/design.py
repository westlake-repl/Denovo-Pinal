import torch 
import os
import logging
from rich.logging import RichHandler
import argparse
from accelerate import Accelerator, InitProcessGroupKwargs
import torch.distributed as dist
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
    parser.add_argument("--multinoimal_temperature", type=float, default=0.3)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    assert not (args.saprot_sample_type == "argmax" and args.multinoimal_temperature != 0.3) , "If you choose argmax, the specific multinoimal_temperature would be of no use."

    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=4))
    accelerator = Accelerator(kwargs_handlers=[timeout_kwargs])
    desc = open(args.desc_path).read().strip()
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

    t2struc, text_tokenizer, structre_tokenizer, saprot, saprot_text_tokenizer, saprot_tokenizer = load_pinal()
    if dist.is_initialized():

        # design_num = args.num // (accelerator.num_processes*5 ) 
        design_num = args.num // accelerator.num_processes 
        rank = accelerator.process_index
    else:
        # design_num = args.num // 5
        design_num = args.num
        rank = 0
    
    print(f"Rank: {rank}, Design number: {design_num}")
    res = PinalDesign(desc, design_num, t2struc, text_tokenizer, structre_tokenizer, saprot, \
                      saprot_text_tokenizer, saprot_tokenizer, args.temperature, args.saprot_sample_type, args.multinoimal_temperature)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # write desc to file
    if dist.is_initialized() and rank == 0:
        with open(os.path.join(output_dir, "desc.txt"), "w") as f:
            f.write(desc)
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