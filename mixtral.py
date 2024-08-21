from transformers import AutoModelForCausalLM, AutoTokenizer

class Mixtral:
    def __init__(self, model_name="mistralai/Mixtral-8x7B-Instruct-v0.2"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate(self, messages):
        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to("cuda")
        generated_ids = self.model.generate(model_inputs, max_new_tokens=1000)
        decoded = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return decoded
    
if __name__ == "__main__":
    # Example usage
    model_name = "mistralai/Mixtral-8x7B-Instruct-v0.2"
    messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
    ]
    mixtral = Mixtral(model_name)
    response = mixtral.generate(messages)
    print(response)
