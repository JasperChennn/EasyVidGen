import torch
from transformers import AutoTokenizer, UMT5EncoderModel

import os
os.environ["TRANSFORMERS_ALLOW_TORCH_LOAD_WITH_UNSAFE_WEIGHTS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

class umT5Embedder:
    # available_models = ["google/t5-v1_1-xxl"]

    def __init__(
        self,
        device,
        from_pretrained=None,
        *,
        cache_dir=None,
        hf_token=None,
        use_text_preprocessing=True,
        t5_model_kwargs=None,
        torch_dtype=None,
        use_offload_folder=None,
        model_max_length=512,
        local_files_only=False,
    ):
        # from_pretrained="google/t5-v1_1-xxl" # zijian
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype or torch.bfloat16
        self.cache_dir = cache_dir

        if t5_model_kwargs is None:
            t5_model_kwargs = {
                "low_cpu_mem_usage": True,
                "torch_dtype": self.torch_dtype,
            }

            if use_offload_folder is not None:
                t5_model_kwargs["offload_folder"] = use_offload_folder
                t5_model_kwargs["device_map"] = {
                    "shared": self.device,
                    "encoder.embed_tokens": self.device,
                    "encoder.block.0": self.device,
                    "encoder.block.1": self.device,
                    "encoder.block.2": self.device,
                    "encoder.block.3": self.device,
                    "encoder.block.4": self.device,
                    "encoder.block.5": self.device,
                    "encoder.block.6": self.device,
                    "encoder.block.7": self.device,
                    "encoder.block.8": self.device,
                    "encoder.block.9": self.device,
                    "encoder.block.10": self.device,
                    "encoder.block.11": self.device,
                    "encoder.block.12": "disk",
                    "encoder.block.13": "disk",
                    "encoder.block.14": "disk",
                    "encoder.block.15": "disk",
                    "encoder.block.16": "disk",
                    "encoder.block.17": "disk",
                    "encoder.block.18": "disk",
                    "encoder.block.19": "disk",
                    "encoder.block.20": "disk",
                    "encoder.block.21": "disk",
                    "encoder.block.22": "disk",
                    "encoder.block.23": "disk",
                    "encoder.final_layer_norm": "disk",
                    "encoder.dropout": "disk",
                }
            else:
                t5_model_kwargs["device_map"] = {
                    "shared": self.device,
                    "encoder": self.device,
                }

        self.use_text_preprocessing = use_text_preprocessing
        self.hf_token = hf_token
        # assert from_pretrained in self.available_models
        self.tokenizer = AutoTokenizer.from_pretrained(
            from_pretrained,
            subfolder="tokenizer",
            model_max_length=model_max_length,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            use_fast=False,  # 核心修改：禁用快速分词器
            trust_remote_code=True  # 可选：如果是自定义 T5 模型
        )
        self.model = UMT5EncoderModel.from_pretrained(
            from_pretrained,
            subfolder="text_encoder",
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            # use_safetensors=False,
            **t5_model_kwargs,
        ).eval()
        self.model_max_length = model_max_length

    def get_text_embeddings(self, texts):
        text_tokens_and_mask = self.tokenizer(
            texts,
            max_length=self.model_max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        input_ids = text_tokens_and_mask["input_ids"].to(self.device)
        attention_mask = text_tokens_and_mask["attention_mask"].to(self.device)
        seq_lens = attention_mask.gt(0).sum(dim=1).long()
        with torch.no_grad():
            text_encoder_embs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )["last_hidden_state"].detach()
        text_encoder_embs = [u[:v] for u, v in zip(text_encoder_embs, seq_lens)]
        text_encoder_embs = torch.stack(
            [torch.cat([u, u.new_zeros(self.model_max_length - u.size(0), u.size(1))]) for u in text_encoder_embs], dim=0
        )
        return text_encoder_embs, attention_mask


if __name__ == "__main__":
    encoder = umT5Embedder(
        from_pretrained="/mnt/dataset/projs/pretrained_models/WoW-1-Wan-1.3B-2M-Diffusers", 
        device='cuda:1')
    text_embeddings = encoder.get_text_embeddings(["Hello, world!"])
    print(f" text_embeddings.shape          = ", text_embeddings[0].shape) # torch.Size([1, 512, 4096])
    input_ids = torch.tensor([[78637,292,312,48694,13706,80959,301,289,9934,280,1753,7868,1,0,0,0,0,0,0,0,0]])
    text_embeddings = encoder.model(input_ids=input_ids.to("cuda:1"), attention_mask=None)["last_hidden_state"].detach()
    print(f" text_embeddings.shape          = ", text_embeddings[0].shape) # torch.Size([21, 4096])
