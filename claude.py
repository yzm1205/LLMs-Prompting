import anthropic
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="./src/.env")


class ClaudeModel:

    def __init__(self, system_role, model="claude-3-opus-20240229"):
      
        self.client = anthropic.Anthropic(api_key=os.getenv("claude_key"))
        self.model = model
        self.sys_role = system_role
        print(f"\n {model} \n")

    def generate_response(self, prompt, max_tokens=1000, temperature=1):
        """
        Generate a response using the Claude model.

        Args:
            prompt (str): The prompt to use for generating the response.
            max_tokens (int): The maximum number of tokens to generate. Defaults to 100.
            temperature (float): The temperature to use for generating the response. Defaults to 0.7.
    
        Returns:
            list: A list of generated responses.
        """
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=self.sys_role,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        )

        return response.content[0].text

if __name__ == "__main__":
    # Example usage
    system_content= f"You are a chat bot assiting people with their queries."
    model = ClaudeModel(system_role=system_content)
    prompt = "what is the purpose of life?"
    response = model.generate_response(prompt)
    print(response)
