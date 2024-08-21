from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" # the device to load the model onto

class Mistral:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate(self, messages):
        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to(device)
        self.model.to(device)
        generated_ids = self.model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
        decoded = self.tokenizer.batch_decode(generated_ids)
        return decoded[0]

if __name__ == "__main__":
    # Example usage
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
    ]
    mistral = Mistral(model_name)
    response = mistral.generate(messages)
    print(response)



