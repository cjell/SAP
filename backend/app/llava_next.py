import torch
from PIL import Image
from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    BitsAndBytesConfig,
)


class LLaVANextCaptioner:
    def __init__(self, model_path="backend/models/llava-next"):

        print("-Loading LLaVA Processor-")
        self.processor = LlavaNextProcessor.from_pretrained(
            model_path,
            use_fast=True
        )

        print("-Configuring 4-bit quantization-")
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        print("-Loading LLaVA Model-")
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=self.bnb_config,
            torch_dtype=torch.bfloat16,   
            device_map="auto"            
        )

        self.model.generation_config.pad_token_id = self.model.config.eos_token_id

        self.prompt_text = (
            "You are a plant identification expert. Describe this plant in a single "
            "coherent paragraph focusing ONLY on biological and morphological traits: "
            "leaf shape, venation, margins, arrangement, stem type, branching, flower "
            "structure, petal count, bracts, inflorescence, colors, and textures. "
            "Do NOT mention photography, background, lighting, or camera details. "
            "Do NOT use bullet points."
        )

        self.max_new_tokens = 200

        print("-LLaVA Loaded-")

    def caption(self, image: Image.Image) -> str:

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.prompt_text},
                ]
            }
        ]

        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        return self.processor.decode(output_ids[0], skip_special_tokens=True).strip()


if __name__ == "__main__":
    img = Image.open("testing/test_images/Aconitum_heterophyllum_test.jpg").convert("RGB")
    llava = LLaVANextCaptioner()
    print(llava.caption(img))
