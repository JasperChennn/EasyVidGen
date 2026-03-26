import torch
import os

os.environ["TRANSFORMERS_ALLOW_TORCH_LOAD_WITH_UNSAFE_WEIGHTS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
from typing import Dict, List, Optional, Tuple

class Qwen3Embedder:
    def __init__(
        self,
        device,
        from_pretrained=None,
        *,
        cache_dir=None,
        hf_token=None,
        use_text_preprocessing=True,
        model_kwargs=None,
        torch_dtype=None,
        use_offload_folder=None,
        model_max_length=512,
        local_files_only=False,
    ):
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype or torch.bfloat16
        self.cache_dir = cache_dir

        if model_kwargs is None:
            model_kwargs = {
                "torch_dtype": self.torch_dtype,
                "device_map": self.device, #"auto",
            }
            if use_offload_folder is not None:
                model_kwargs["offload_folder"] = use_offload_folder
                # 可按需配置 device_map 做部分 offload

        self.use_text_preprocessing = use_text_preprocessing
        self.hf_token = hf_token

        print("=============== Loading processor ===============")
        self.processor = AutoProcessor.from_pretrained(
            from_pretrained,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        print("=============== Loading model ===============")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            from_pretrained,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            **model_kwargs,
        ).eval()
        print("=============== Model loaded ===============")
        self.model.requires_grad_(False)
        self.model_max_length = getattr(
            self.processor.tokenizer,
            "model_max_length",
            model_max_length,
        )
        if self.model_max_length is None or self.model_max_length > 1e6:
            self.model_max_length = model_max_length

    def _messages_from_texts(self, text_or_texts: str | list[str], images: Optional[Image.Image] = None):
        """支持 (text, image) 或 (texts 列表)。返回一条 conversation 或 list of conversations。"""
        if isinstance(text_or_texts, list):
            return [[{"role": "user", "content": [{"type": "text", "text": t}]}] for t in text_or_texts]
        text = text_or_texts
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            return [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": image} for image in images] +
                    [{"type": "text", "text": text}],
                }
            ]
        return [{"role": "user", "content": [{"type": "text", "text": text}]}]

    def _process_vlm_inputs_to_tokens(self, vlm_inputs, B: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[list], torch.Tensor]:
        """Convert VLM inputs to tokens.

        Returns:
            Tuple of (inputs_embeds, attention_mask, visual_pos_masks, deepstack_image_embeds, position_ids)
        """
        # Handle both old format (List[Dict]) and new format (Dict[str, Tensor])
        if isinstance(vlm_inputs, list):
            # Old format: List[Dict] - do padding and batching
            input_ids_list = [vlm_input['input_ids'] for vlm_input in vlm_inputs]
            attention_mask_list = [vlm_input.get('attention_mask') for vlm_input in vlm_inputs]
            pixel_values_list = [vlm_input.get('pixel_values') for vlm_input in vlm_inputs]
            image_grid_thw_list = [vlm_input.get('image_grid_thw') for vlm_input in vlm_inputs]

            # Pad input_ids and attention_mask to same length
            max_seq_len = max(ids.shape[1] for ids in input_ids_list)
            padded_input_ids = []
            padded_attention_masks = []
            
            for ids, mask in zip(input_ids_list, attention_mask_list):
                if ids.shape[1] < max_seq_len:
                    padding_size = max_seq_len - ids.shape[1]
                    # Pad input_ids with zeros
                    id_padding = torch.zeros(ids.shape[0], padding_size, dtype=ids.dtype, device=ids.device)
                    padded_ids = torch.cat([ids, id_padding], dim=1)
                    # Pad attention_mask with zeros (padding tokens should be ignored)
                    mask_padding = torch.zeros(mask.shape[0], padding_size, dtype=mask.dtype, device=mask.device)
                    padded_mask = torch.cat([mask, mask_padding], dim=1)
                else:
                    padded_ids = ids
                    padded_mask = mask
                padded_input_ids.append(padded_ids)
                padded_attention_masks.append(padded_mask)

            # Batch process
            input_ids_batch = torch.cat(padded_input_ids, dim=0).to(self.device)
            attention_mask_batch = torch.cat(padded_attention_masks, dim=0).to(self.device)
            pixel_values_batch = torch.cat([pv.to(self.device) for pv in pixel_values_list], dim=0)
            image_grid_thw_batch = torch.cat([igt.to(self.device) for igt in image_grid_thw_list], dim=0)
        else:
            # New format: Dict[str, Tensor] - already batched and padded by collate_fn
            input_ids_batch = vlm_inputs['input_ids'].to(self.device)
            attention_mask_batch = vlm_inputs['attention_mask'].to(self.device)
            pixel_values_batch = vlm_inputs['pixel_values'].to(self.device)
            image_grid_thw_batch = vlm_inputs['image_grid_thw'].to(self.device)

        # Get input embeddings
        inputs_embeds = self.model.get_input_embeddings()(input_ids_batch)

        # Process images - handle different return formats between Qwen2.5-VL and Qwen3-VL
        image_embeds, deepstack_image_embeds = self.model.get_image_features(pixel_values_batch, image_grid_thw_batch)

        image_embeds = torch.cat(image_embeds, dim=0).to(self.device, self.torch_dtype)

        # Insert image embeddings
        image_mask, _ = self.model.model.get_placeholder_mask(
            input_ids_batch, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        visual_pos_masks = image_mask[..., 0]  # [B, seq_len] - visual positions only

        # Compute position_ids (position_ids remains as original: [3, B, seq_len])
        # Qwen3-VL get_rope_index has different signature: (input_ids, image_grid_thw, video_grid_thw, attention_mask)
        position_ids, _rope_deltas = self.model.model.get_rope_index(
            input_ids=input_ids_batch,
            image_grid_thw=image_grid_thw_batch,
            video_grid_thw=None,  # No video in current implementation
            attention_mask=attention_mask_batch
        )

        return inputs_embeds, attention_mask_batch, visual_pos_masks, deepstack_image_embeds, position_ids
    

    @torch.no_grad()
    def extract_und_features(
        self,
        vlm_inputs
    ) -> torch.Tensor:
        """Extract understanding features from VLM last layer."""
        if isinstance(vlm_inputs, list):
            B = len(vlm_inputs)
        else:
            B = vlm_inputs['input_ids'].shape[0]

        # Returns: inputs_embeds, attention_mask, visual_pos_masks, deepstack_image_embeds, position_ids
        inputs_embeds, attention_mask, visual_pos_masks, deepstack_image_embeds, position_ids = self._process_vlm_inputs_to_tokens(vlm_inputs, B)

        # Forward through VLM with proper attention_mask and DeepStack features
        vlm_kwargs = {
            'inputs_embeds': inputs_embeds,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'past_key_values': None,
            'use_cache': False,
            'output_attentions': False,
            'output_hidden_states': True,
            'return_dict': True
        }

        # Add DeepStack parameters for Qwen3-VL
        if visual_pos_masks is not None:
            vlm_kwargs['visual_pos_masks'] = visual_pos_masks
        if deepstack_image_embeds is not None:
            vlm_kwargs['deepstack_visual_embeds'] = deepstack_image_embeds

        with torch.no_grad():
            vlm_output = self.model.model.language_model(**vlm_kwargs)

        # Extract last layer features directly
        last_layer_features = vlm_output.hidden_states[-1]  # [B, seq_len, vlm_dim]
        return last_layer_features 


    def _preprocess_vlm_messages(self, instruction: str, images: Optional[Image.Image] = None) -> Dict[str, torch.Tensor]:
        messages = self._messages_from_texts(instruction, images)
        text = self.processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        encoded = self.processor(text=[text], images=images, return_tensors='pt')
        vlm_inputs = {
            'input_ids': encoded['input_ids'].to(self.device),
            'attention_mask': encoded['attention_mask'].to(self.device), 
            'pixel_values': encoded['pixel_values'].to(self.device),
            'image_grid_thw': encoded.get('image_grid_thw', None)
        }
        if vlm_inputs['image_grid_thw'] is not None:
            vlm_inputs['image_grid_thw'] = vlm_inputs['image_grid_thw'].to(self.device)
        return vlm_inputs

    @torch.no_grad()
    def get_answer(
        self,
        messages_list,
        *,
        max_new_tokens=128,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        **generate_kwargs,
    ):
        """
        根据 messages 生成回复文本。支持纯文本或多模态（文本+图像）。

        Args:
            messages_list: 列表，每个元素为一轮对话的 messages（与 chat 模板一致）。
                例如 [{"role": "user", "content": [{"type": "text", "text": "Hello."}]}]
                或含 image 的 [{"type": "image", "image": url_or_pil}, {"type": "text", "text": "Describe this."}]
            max_new_tokens: 最大生成 token 数。
            do_sample: 是否采样；False 为贪心解码。
            temperature: 采样温度（do_sample=True 时有效）。
            top_p: nucleus 采样参数。
            **generate_kwargs: 透传给 model.generate 的其他参数。

        Returns:
            list[str]: 每条 message 对应的生成文本。
        """
        if not messages_list:
            return []
        # 兼容单条：传入一条 message 时包装成 list of list
        if isinstance(messages_list[0], dict) and "role" in messages_list[0]:
            messages_list = [messages_list]

        batch_inputs = []
        for messages in messages_list:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            batch_inputs.append(inputs)

        # 按 batch 拼接或逐条生成（若长度不一则逐条更稳妥）
        input_ids = torch.cat([x["input_ids"] for x in batch_inputs], dim=0)
        attention_mask = torch.cat(
            [
                x.get("attention_mask", torch.ones_like(x["input_ids"]))
                for x in batch_inputs
            ],
            dim=0,
        )
        input_ids = input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)

        # pixel_values 等若存在则一并移到 device
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        for k in batch_inputs[0].keys():
            if k in ("input_ids", "attention_mask"):
                continue
            if all(k in x and isinstance(x.get(k), torch.Tensor) for x in batch_inputs):
                model_inputs[k] = torch.cat([x[k] for x in batch_inputs], dim=0).to(
                    self.model.device
                )

        gen_kw = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.processor.tokenizer.pad_token_id
            or self.processor.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kw["temperature"] = temperature
            gen_kw["top_p"] = top_p
        gen_kw.update(generate_kwargs)

        generated_ids = self.model.generate(**model_inputs, **gen_kw)

        # 只保留新生成部分
        input_lens = attention_mask.sum(dim=1).long()
        generated_trimmed = [
            out_ids[in_len:].tolist()
            for out_ids, in_len in zip(generated_ids, input_lens)
        ]
        output_texts = self.processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_texts

    @torch.no_grad()
    def get_answer_from_text(self, prompts, **kwargs):
        """
        纯文本问答的便捷接口。每条 prompt 对应一个回复。

        Args:
            prompts: 字符串或字符串列表，用户问题。
            **kwargs: 透传给 get_answer 的参数（如 max_new_tokens, do_sample 等）。

        Returns:
            list[str]: 每个问题对应的生成答案。
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        messages_list = [
            [{"role": "user", "content": [{"type": "text", "text": p}]}]
            for p in prompts
        ]
        return self.get_answer(messages_list, **kwargs)


if __name__ == "__main__":
    model_id = "/mnt/dataset/datasets/cjt_personal/pretrained_models/Qwen3-VL-2B-Instruct/"
    encoder = Qwen3Embedder(
        from_pretrained=model_id,
        device="cuda:0",
    )
    # 测速
    # from time import time
    # t = time()
    # for i in range(10):
    #     vlm_inputs = encoder._preprocess_vlm_messages("Describe this image in Chinese.", image="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg")
    #     text_embeddings = encoder.extract_und_features(vlm_inputs=vlm_inputs)
    #     print("text_embeddings.shape =", text_embeddings.shape)
    # print("time =", time() - t)

    img1 = Image.open("/mnt/dataset/projs/projects/RoboTwin/policy/RDT/debug/debug.png")
    img2 = Image.open("/mnt/dataset/projs/projects/RoboTwin/policy/RDT/debug/debug.png")
    img3 = Image.open("/mnt/dataset/projs/projects/RoboTwin/policy/RDT/debug/debug.png")
    # 假设有多条指令和对应图片
    instructions = [
        "Describe this image in Chinese.",
        "Describe this image in English.",
        "What is in this picture?",
    ]
    images = [img1, img2, img3]  # 对应的 PIL.Image 或其它可被 processor 接受的图像对象

    # 逐条用 _preprocess_vlm_messages 得到单条 vlm_inputs
    vlm_inputs_batch = [
        encoder._preprocess_vlm_messages(instr, images=img)
        for instr, img in zip(instructions, images)
    ]

    # 直接传给 extract_und_features（内部会做 padding / batching）
    features = encoder.extract_und_features(vlm_inputs=vlm_inputs_batch)
    print(features.shape)  # [B, seq_len, vlm_dim]

    # 纯文本问答
    answers = encoder.get_answer_from_text("What is 2+2? Reply in one word.", max_new_tokens=32)
    print("get_answer_from_text:", answers)

    # 多轮 messages 调用
    messages = [[{"role": "user", "content": [{"type": "text", "text": "Say hello in Chinese."}]}]]
    answers = encoder.get_answer(messages, max_new_tokens=64)
    print("get_answer:", answers)

    messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "/mnt/dataset/projs/projects/RoboTwin/policy/RDT/debug/debug.png",
                },
                {"type": "text", "text": "Describe this image in Chinese."},
            ],}
    ]
    answers = encoder.get_answer(messages, max_new_tokens=64)
    print("get_answer:", answers)