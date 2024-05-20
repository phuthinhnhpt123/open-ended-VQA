from peft import LoraConfig, get_peft_model,TaskType 
import torch.nn as nn
from transformers import GPT2Tokenizer
from transformers import AutoModelForCausalLM

from prefix_mappers import MLP

class VQAModel(nn.Module):
    def forward(self, prefix, tokens, mask, q_len, batch_size):
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)

        embedding = self.gpt.transformer.wte(tokens)

        for b in range(batch_size):
            # insert the visual prefix after the question 
            embedding[b,q_len[b]:q_len[b]+self.prefix_length,:] = prefix_projections[b]  
        return self.gpt(inputs_embeds=embedding, attention_mask=mask)
    def generate(self, prefix, tokens,_, q_len):
        prefix_projections = self.clip_project(prefix.view(1, -1)).view(self.prefix_length, self.gpt_embedding_size)

        embedding_txt = self.gpt.transformer.wte(tokens)
        
        embedding_txt[q_len:q_len+self.prefix_length,:] = prefix_projections
        return embedding_txt
    
    def gen_answer(self, prefix, tokens, q_len,max_len):
        prefix_projections = self.clip_project(prefix.view(1, -1)).view(self.prefix_length, self.gpt_embedding_size)

        embedding_txt = self.gpt.transformer.wte(tokens)
        
        embedding_txt[q_len:q_len+self.prefix_length,:] = prefix_projections

        inputs_embeds = embedding_txt.view(1,tokens.size(0),-1)

        # print('input_embed: ',inputs_embeds)
        # print('input_embed shape: ',inputs_embeds.shape)

        outputs = self.gpt.generate(
            inputs_embeds=inputs_embeds,
            num_beams=5,
            max_new_tokens=max_len,
            num_return_sequences=1,
            no_repeat_ngram_size=1,
            remove_invalid_values=True,
        )

        return self.tokenizer.decode(outputs[0],skip_special_tokens=True)
    def __init__(
        self,
        prefix_length=2,
        prefix_size=512,
        setting="lora",
        mapping_type="MLP",
        args=None,
    ):
        super(VQAModel, self).__init__()
        gpttype = args.model_type
        self.gpttype = gpttype
        self.setting = setting
        self.prefix_length = prefix_length
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpttype)
        self.gpt = AutoModelForCausalLM.from_pretrained(gpttype,load_in_8bit=True,device_map='auto')

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.gpt.generation_config.pad_token_id = self.tokenizer.encode(self.tokenizer.pad_token)[0]
        # load the relevant fine-tuning strategy 

        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
        self.gpt = get_peft_model(self.gpt,peft_config)
        self.gpt.print_trainable_parameters()

        
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if mapping_type == "MLP":
            self.clip_project = MLP((
                    prefix_size,
                    (self.gpt_embedding_size * prefix_length) // 2,
                    self.gpt_embedding_size * prefix_length))
        else:
            raise ValueError("invalid mapping type")