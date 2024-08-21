import google.generativeai as palm
import base64
import json
import pprint

palm.configure(api_key="YOUR API KEY")
# These parameters for the model call can be set by URL parameters.
model = "" # @param {isTemplate: true}
temperature = 0.7 # @param {isTemplate: true}
candidate_count = 1 # @param {isTemplate: true}
top_k = 40 # @param {isTemplate: true}
top_p = 0.95 # @param {isTemplate: true}
max_output_tokens = 1024 # @param {isTemplate: true}
text_b64 = "" # @param {isTemplate: true}
stop_sequences_b64 = ""  # @param {isTemplate: true}
safety_settings_b64 = ""  # @param {isTemplate: true}

# Convert the prompt text param from a bae64 string to a string.
text = base64.b64decode(text_b64).decode("utf-8")

# Convert the stop_sequences and safety_settings params from base64 strings to lists.
stop_sequences = json.loads(base64.b64decode(stop_sequences_b64).decode("utf-8"))
safety_settings = json.loads(base64.b64decode(safety_settings_b64).decode("utf-8"))

defaults = {
  'model': model,
  'temperature': temperature,
  'candidate_count': candidate_count,
  'top_k': top_k,
  'top_p': top_p,
  'max_output_tokens': max_output_tokens,
  'stop_sequences': stop_sequences,
  'safety_settings': safety_settings,
}

# Show what will be sent with the API call.
pprint.pprint(defaults | {'prompt': text})


# Call the model and print the response.
response = palm.generate_text(
  **defaults,
  prompt=text
)
print(response.candidates[0]['output'])