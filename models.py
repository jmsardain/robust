from openai import OpenAI
import os

class OpenAIClient:
    def __init__(self, model="gpt-4o-mini", temperature=0.0):
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a text-to-SQL model. Output only SQL."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature
        )
        return resp.choices[0].message.content.strip()



class QwenClient:
    def __init__(self, model="Qwen/Qwen-2.5B-Coder-Instruct", temperature=0.0):
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            device_map="cuda",
            torch_dtype="auto"
        )
        self.model.generation_config.temperature=None
        self.model.generation_config.top_p=None
        self.model.generation_config.top_k=None
               
    def generate(self, prompt: str) -> str:
        system_instruction = "You are a text-to-SQL model. Output only SQL.\n"
        full_prompt = system_instruction + prompt

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )
        generated = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated[len(full_prompt):].strip()
