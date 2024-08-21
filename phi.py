import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.random.manual_seed(0)

class phi3mini:
    def __init__(self,system_role,cuda_device=0):
        self.system_role = system_role
        self.cuda = cuda_device
        
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-128k-instruct", 
            device_map=f'cuda:{self.cuda}', 
            torch_dtype="auto", 
            trust_remote_code=True, 
        )
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def generate_response(self, prompt, max_tokens=500, temperature=1):
        messages = [
            {"role": "system", "content": self.system_role},
            {"role": "user", "content": prompt},
        ]
        generation_args = {
            "max_new_tokens": max_tokens,
            "return_full_text": False,
            "temperature": temperature,
            "do_sample": False,
        }

        output = self.pipe(messages, **generation_args)
        # print(output[0]['generated_text'])
        return output[0]['generated_text']
    
if __name__ == "__main__":
  
    system_content= f"You are a chat bot assiting people with their queries."
    model = phi3mini(system_role=system_content)
    prompt = "What is the purpose of life?"
    generated_text = model.generate_response(prompt)
    print(generated_text)