from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class LocalLLM:
    def __init__(self, model_id="mistralai/Mistral-7B-Instruct-v0.1", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.7,
            return_full_text=False
        )

    def __call__(self, prompt: str) -> str:
        result = self.pipeline(prompt)
        return result[0]['generated_text']
