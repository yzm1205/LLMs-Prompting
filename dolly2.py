from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class Dolly2:
    def __init__(self, baseModel, load_8bit=True):
        self.tokenizer = AutoTokenizer.from_pretrained(baseModel)
        self.model = AutoModelForCausalLM.from_pretrained(baseModel, load_in_8bit=load_8bit, torch_dtype=torch.float16, device_map="auto")
        

    def generate(self, text):
        generator = pipeline(task='text-generation', model=self.model, tokenizer=self.tokenizer)
        return generator(text)
  
if __name__ == "__main__":
    # Example usage
    baseModel = "databricks/dolly-v2-12b"
    load_8bit = True  
    dolly = Dolly2(baseModel, load_8bit)
    response = dolly.generate("Python code to remove duplicates from dataframe")
