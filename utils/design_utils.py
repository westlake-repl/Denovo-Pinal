import torch 
import os
from omegaconf import OmegaConf
from transformers import AutoTokenizer, EsmTokenizer
from collections import OrderedDict
from transformers import GenerationConfig
import torch.nn as nn
import logging
from rich.logging import RichHandler
import math
from models import StructureTokenPredictionModel, SaProtIFModel

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

logger = logging.getLogger("rich")
T2struc_NAMES=[
    "T2struc-1.2B", 
    "T2struc-15B",
]
SAPROT_NAMES=[
    "SaProtT"
]
MODEL_ROOT = os.environ.get("PINAL_MODEL_ROOT", "weights")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T2STRUCT = os.environ.get("DEFAULT_MODEL_T2STRUCT", "pytorch_model.bin")
T2struc_NAME = os.environ.get("T2struc_NAME", "T2struc-1.2B")

TO_LIST = lambda seq: [
            seq[i, ...].detach().cpu().numpy().tolist() for i in range(seq.shape[0])
        ]

def load_T2Struc(cfg, model_name):
    model =  StructureTokenPredictionModel(cfg.model).to(torch.bfloat16) # Since we train the model with bfloat16
    model.load_state_dict(torch.load(os.path.join(MODEL_ROOT, model_name, T2STRUCT), map_location='cpu'))
    return model.to(DEVICE)
    
    
def load_T2Struc_tokenizers(cfg):
    text_tokenizer =  AutoTokenizer.from_pretrained(cfg.model["lm"])
    structure_tokenizer = EsmTokenizer.from_pretrained(cfg.model["tokenizer"])
    return text_tokenizer, structure_tokenizer


def load_SaProtT(cfg, SaProt_name="SaProtT"):
    model = SaProtIFModel(cfg.model).to(torch.bfloat16)
    ckpt = torch.load(os.path.join(MODEL_ROOT, SaProt_name, "pytorch_model.bin"), map_location='cpu')
    ckpt_model = OrderedDict({k[len("module."):]: v for k, v in ckpt["state_dict"].items()})
    model.load_state_dict(ckpt_model, strict=False)
    return model.to(DEVICE)


def load_SaProt_tokenizers(cfg):
    SaProt_tokenizer = EsmTokenizer.from_pretrained("checkpoints/SaProt")
    text_tokenizer = AutoTokenizer.from_pretrained("checkpoints/microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract")
    return SaProt_tokenizer, text_tokenizer
    

def load_T2Struc_and_tokenizers():
    assert T2struc_NAME in T2struc_NAMES
    logger.info("T2strcu Model: " + T2struc_NAME)
    cfg = OmegaConf.load(os.path.join(MODEL_ROOT, T2struc_NAME, "config.yaml"))
    model = load_T2Struc(cfg, T2struc_NAME)
    model = model.eval()
    text_tokenizer, structre_tokenizer = load_T2Struc_tokenizers(cfg)
    return model, text_tokenizer, structre_tokenizer

def load_SaProtT_and_tokenizers(SaProt_name="SaProtT"):
    assert SaProt_name in SAPROT_NAMES
    cfg = OmegaConf.load(os.path.join(MODEL_ROOT, SaProt_name, "config.yaml"))
    model = load_SaProtT(cfg, SaProt_name)
    model = model.eval()
    SaProt_tokenizer, text_tokenizer = load_SaProt_tokenizers(cfg)
    return model, text_tokenizer, SaProt_tokenizer


def load_pinal():
    logger.info("Loading Pinal...")
    global t2struc, text_tokenizer, structre_tokenizer, saprot, saprot_text_tokenizer, saprot_tokenizer
    
    t2struc, text_tokenizer, structre_tokenizer = load_T2Struc_and_tokenizers()
    saprot, saprot_text_tokenizer, saprot_tokenizer = load_SaProtT_and_tokenizers()
    logger.info("Pinal loaded successfully.")

def T2StrcuDefaultGenerationConfig():
    return GenerationConfig(
        temperature=1,
        top_k=40,
        top_p=1,
        do_sample=True,
        num_beams=1,
        repetition_penalty=1.0,
        max_length=1024,
        min_length=64,
    )

def T2StrucPrepareGenerationInputs(t2struc, desc, text_tokenizer, structre_tokenizer):
    batch = {}
    desc_encodings = text_tokenizer(
        desc,
        return_tensors="pt",
        max_length=768,
        truncation=True,
        padding="longest",
    )
    batch["text_ids"] = desc_encodings.input_ids.to(DEVICE)
    batch["text_masks"] = desc_encodings.attention_mask.to(DEVICE)
    text_hidden_states, text_attention_mask = t2struc.infer_text(batch)
    
    start_id = structre_tokenizer.cls_token_id
    stop_id = structre_tokenizer.eos_token_id
    pad_id = structre_tokenizer.pad_token_id
    input_ids = (torch.zeros((1)) + start_id).unsqueeze(0).to(torch.long).to(DEVICE) # create batch dim
    
    return {
        "input_ids": input_ids,
        "bos_token_id": start_id,
        "eos_token_id": stop_id,
        "pad_token_id": pad_id, 
        "encoder_hidden_states": text_hidden_states,
        "encoder_attention_mask": text_attention_mask,
    }
    
def T2StrucGeneration(t2struc, T2StrucGenerationDict, T2StrucGenerationConfig, structre_tokenizer):
    sample_results = t2struc.plm.generate(
        do_sample=True,
        generation_config=T2StrucGenerationConfig,
        num_return_sequences=5,
        return_dict_in_generate=True, 
        **T2StrucGenerationDict
    )
    return {
        "structure": structre_tokenizer.batch_decode(TO_LIST(sample_results.sequences)),
        "structrue_logp": torch.sum(sample_results.log_p, dim=-1).cpu().numpy().tolist(),
    }

def SaProtPrepareGenerationInputs(structures, desc, saprot_text_tokenizer, saprot_tokenizer):
    saprot_input = []
    for structrue in structures:
        saprot_input.append("#" + "#".join([i for i in structrue.split() if i.isalpha() or i == "#"]))
    inputs = saprot_tokenizer(saprot_input, return_tensors="pt", max_length=1024, truncation=True, padding="max_length")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    encodings = saprot_text_tokenizer(
        desc,
        padding="longest", # max_length,longest
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    saprot_text_ids = encodings["input_ids"].to(DEVICE)
    saprot_text_masks= encodings["attention_mask"].to(DEVICE)
    return {
            "mask_prot_ids": inputs["input_ids"],
            "prot_masks": inputs["attention_mask"],
            'text_ids': saprot_text_ids,
            'text_masks': saprot_text_masks, 
            "prot_ids": None
        }

def SaProtGeneration(saprot, SaProtInputDict):
    with torch.no_grad():
        out = saprot(SaProtInputDict)
    logits = out["outputs"].logits
    predicted_token = logits.argmax(-1)
    probs = nn.functional.softmax(logits, dim=-1)
    predicted_probs = torch.gather(probs, dim=-1, index=predicted_token.unsqueeze(-1)).squeeze()
    sequence_log_p = torch.sum(predicted_probs.log() * SaProtInputDict["prot_masks"], dim=-1).to(torch.float32)
    
    predicted_token_list = list(predicted_token)
    predicted_token_list = [saprot_tokenizer.convert_ids_to_tokens(i) for i in predicted_token_list]
    sequence_token_list = []
    for i, predicted_token in enumerate(predicted_token_list):
        cur_list = []
        for i in predicted_token[1:]:
            if i[0] == "<":
                break
            cur_list.append(i[0])
        sequence_token_list.append(cur_list)

    sequence_token_list = ["".join(sequence_token) for sequence_token in sequence_token_list]

    return {
        "sequence": sequence_token_list,
        "sequence_logp": sequence_log_p.cpu().numpy().tolist(),
    }


def deduplicate_tuples(tuple_list, keyid=0):
    seen = set()
    deduplicated_list = []
    
    for tpl in tuple_list:
        key = (tpl[keyid])
        if key not in seen:
            seen.add(key)
            deduplicated_list.append(tpl)
    
    return deduplicated_list

def desc_sanity_check(desc):
    if not desc.endswith("."):
        logger.info("The description should end with a period.")
        desc += "."
    return desc

def PinalDesign(desc, num):
    logger.info("Start designing...")
    global t2struc, text_tokenizer, structre_tokenizer, saprot, saprot_text_tokenizer, saprot_tokenizer
    assert t2struc and text_tokenizer and structre_tokenizer and saprot and saprot_text_tokenizer and saprot_tokenizer

    desc = desc_sanity_check(desc)
    structures, structures_logp, sequences, sequences_logp = [], [], [], []
    
    # prepare input
    T2StrucGenerationDict = T2StrucPrepareGenerationInputs(t2struc, desc, text_tokenizer, structre_tokenizer)
    T2StrucGenerationConfig = T2StrcuDefaultGenerationConfig()
    for i in range(0, 100, math.ceil(500 / num)):
        ## t2struc generation
        t2struc_res = T2StrucGeneration(t2struc, T2StrucGenerationDict, T2StrucGenerationConfig, structre_tokenizer)
        structures.extend(t2struc_res["structure"])
        structures_logp.extend(t2struc_res["structrue_logp"])
        
        ## SaProt-T
        SaProtInputDict = SaProtPrepareGenerationInputs(t2struc_res["structure"], desc, saprot_text_tokenizer, saprot_tokenizer)
        saprot_res = SaProtGeneration(saprot, SaProtInputDict)        
        sequences.extend(saprot_res["sequence"])
        sequences_logp.extend(saprot_res["sequence_logp"])
        logger.info(f"{i + math.ceil(500 / num)} % Sequence Designed Done")

    ## rebatch
    res = list(zip(structures, structures_logp, sequences, sequences_logp))
    res = deduplicate_tuples(res, keyid=2) # deduplicate by sequence
    res = sorted(res, key=lambda x: (x[1] + x[3]) / len(x[2]) , reverse=True)  # sort by normed logp
    # transfer the list of tuple to list of dict
    res = [i[2] for i in res]
    
    return res

def protrek_score(seq, pred_text, seq_type='prot'):
    global protrek
    torch.cuda.empty_cache()
    if not isinstance(seq, list):
        seq = [seq]
    if not isinstance(pred_text, list):
        pred_text = [pred_text]

    with torch.no_grad():
        if seq_type == 'prot':
            seq_repr = protrek.get_protein_repr(seq)
        else:
            seq_repr = protrek.get_structure_repr(seq)
        pred_text_repr = protrek.get_text_repr(pred_text)

        sim_pred_text = torch.matmul(pred_text_repr, seq_repr.T) / protrek.temperature

        sim_pred_mask = torch.eye(sim_pred_text.size(0), device=sim_pred_text.device)
        sim_pred_text = sim_pred_text.masked_fill(sim_pred_mask == 0, -1e9)
        sim_pred_text = sim_pred_text.max(dim=1)[0]
    return sim_pred_text

def load_protrek():
    import sys
    sys.path.append("/daifengyuan/my_project/analysis/clip_score")
    from model.claprot.claprot_trimodal_model import CLAProtTrimodalModel
    global protrek
    # loading model
    model_config = {
        "protein_config": "/daifengyuan/ProteinQA/Models/esm2_t12_35M_UR50D",
        "text_config": "/daifengyuan/ProteinQA/Models/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "structure_config": "/daifengyuan/ProteinQA/Models/foldseek_t12_35M",
        "from_checkpoint": "/daifengyuan/ProteinQA/Models/CLAProt_35M_uniref50_high_sim.pt"
    }
    protrek = CLAProtTrimodalModel(**model_config)
    protrek.to("cuda").eval()
    logger.info("Protrek loaded successfully.")