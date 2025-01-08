import random
from  models import ABSmodule
from models.gpt2 import GPT2LMHeadModel, GPT2Config
from transformers import T5EncoderModel, T5Config


class StructureTokenPredictionModel(ABSmodule):
    def __init__(self, model_config):
        super().__init__()
        self.config = model_config
        
        # ===================== Text Encoder ===================== #
        lm_config = T5Config.from_pretrained(model_config["lm"], local_files_only=True)
        self.lm = T5EncoderModel(lm_config)
        try:
            model_config["lm_emb_dim"] = self.lm.config.hidden_size
        except:
            model_config["lm_emb_dim"] = self.lm.config.d_model
    
        # ===================== Protein Sequence Encoder With Textual Adapter ===================== #
        plm_config = GPT2Config.from_pretrained(model_config["plm_type"], add_cross_attention=True, 
                    cache_dir='checkpoints', local_files_only=True, cross_attention_dim=model_config.lm_hidden_size
                    )
        self.plm = GPT2LMHeadModel(plm_config)
        # set config.hidden_sizes to make deepspeed stage3 stage3_prefetch_bucket_size and other parameters can be set as "auto"
        self.config.hidden_size = self.plm.config.hidden_size
        
        
    def infer(
        self,
        batch,
        text_hidden_states=None,
        text_attention_mask=None,
        return_dict=True,
    ):
        outputs = self.plm(
            input_ids=batch["structure_token_ids"],
            attention_mask=batch["structure_token_masks"],
            encoder_hidden_states=text_hidden_states,
            encoder_attention_mask=text_attention_mask,
            labels=batch["structure_token_ids"],
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
        return text_output.last_hidden_state, batch["text_masks"]


    def forward(self, batch):
        ret = dict()
        ret["text_hidden_states"], ret["text_attention_mask"] = self.infer_text(
            batch
        )
        ret.update(
            self.infer(
                batch,
                text_hidden_states=ret["text_hidden_states"],
                text_attention_mask=ret["text_attention_mask"],
            )
        )
        return ret

