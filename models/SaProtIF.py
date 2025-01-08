import random
import torch
from  models import ABSmodule
from models.esm2 import EsmForMaskedLM
from transformers import BertModel, BertConfig, EsmConfig
from collections import OrderedDict

class NullLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def exception(self, *args, **kwargs):
        pass

class SaProtIFModel(ABSmodule):
    def __init__(self, model_config):
        super().__init__()
        self.config = model_config

        # ===================== Text Encoder ===================== #
        lm_config = BertConfig.from_pretrained(model_config["lm"], local_files_only=True)
        self.lm = BertModel(lm_config)
        
        # ===================== Protein Sequence Encoder With Textual Adapter ===================== #
        plm_config = EsmConfig.from_pretrained(model_config["plm"], local_files_only=True)
        self.plm = EsmForMaskedLM(plm_config)


    def infer(
        self,
        batch,
        text_hidden_states=None,
        text_attention_mask=None,
        return_dict=True,
    ):
        # concat the embeddings of text and token 
        if text_hidden_states.shape[0] != batch["mask_prot_ids"].shape[0]:
            text_hidden_states = text_hidden_states.repeat(batch["mask_prot_ids"].shape[0], 1)
        outputs = self.plm(
            input_ids=batch["mask_prot_ids"],
            attention_mask=batch["prot_masks"],
            text_token_embeddings=text_hidden_states,
            labels=batch["prot_ids"],
            return_dict=return_dict,
            output_hidden_states=True,
        )

        return {"outputs": outputs}

    def infer_text(
        self,
        batch,
    ):
        text_output = self.lm(
            input_ids=batch["text_ids"],
            attention_mask=batch["text_masks"],
            output_hidden_states=True,
        )
        # return text_output.pooler_output, batch["text_masks"]
        return text_output.pooler_output


    def forward(self, batch):
        ret = dict()
        ret["text_hidden_states"] = self.infer_text(batch)
        ret.update(
            self.infer(
                batch,
                text_hidden_states=ret["text_hidden_states"],
            )
        )
        return ret

