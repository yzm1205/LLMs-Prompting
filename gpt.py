import os
import json
from openai import OpenAI
import yaml
from argparse import ArgumentParser
from dotenv import load_dotenv
load_dotenv(dotenv_path="./src/.env")


class GPTModel:
    
    def __init__(self,system_role,model_name='gpt-4'):
    
        self.client = OpenAI(api_key=os.getenv("gpt3_key"))
        self.model_name = model_name
        self.sys_role = system_role
        print(model_name)
        print()
        

    def generate_response(self, prompt, max_tokens=256, temperature=1, top_p=1.0):
        """
        Generate text using the GPT-3 model.

        Args:
            prompt (str): The prompt to use for text generation.
            max_tokens (int): The maximum number of tokens to generate. Defaults to 100.
            temperature (float): The temperature to use for text generation. Defaults to 0.7.
            top_p (float): The top-p value to use for text generation. Defaults to 1.0.
            n (int): The number of completions to generate. Defaults to 1.

        Returns:
            list: A list of generated text outputs.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", 
                    "content": self.sys_role},
                    {"role":"user",
                    "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=0,
            presence_penalty=0
        )

        return response.choices[0].message.content


if __name__ == "__main__":
# Example usage
    
    system_content= f"You are a chat bot assiting people with their queries."
    model = GPTModel(system_role=system_content)
    prompt = "What happened to voyger 1 spacecraft?"
    generated_text = model.generate_response(prompt)
    print(generated_text)