import google.generativeai as genai
import os
from dotenv import load_dotenv
from sympy import Ge
load_dotenv(dotenv_path="./src/.env")


helper  ="https://ai.google.dev/tutorials/python_quickstart"

class GeminiModel:
    def __init__(self,system_role,model = "gemini-pro"):
        genai.configure(api_key=os.getenv("gemini_key"))
        self.model = genai.GenerativeModel(model)
        self.sys_role = system_role
        print(f"Gemini-{model}")

    def generate_response(self,prompt):
        response = self.model.generate_content(self.sys_role + prompt,
                                               generation_config={'temperature': 1, 'max_output_tokens': 1000})
        return response.text
    
if __name__ == "__main__":
    
    system_content= f"You are a chat bot assiting people with their queries."
    model = GeminiModel(system_role=system_content)
    response = model.generate_response("What is purpose of life?")

    print(response)