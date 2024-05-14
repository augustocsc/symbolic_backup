# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import prefix_to_infix
import numpy as np

opt_dict={'sqrt':'np.sqrt','sin':'np.sin','cos':'np.cos','tan':'np.tanh'}

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("augustocsc/gpt-base")

# Prompt the letter 'y' to the model and print the result
input_text = "y"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
output = model.generate(input_ids, max_length=50, 
                        num_return_sequences=1,
                        do_sample=True,
                        pad_token_id=50256)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)

gen_expr = generated_text.replace("y ", "").split(" ")
print(gen_expr)
expr = prefix_to_infix(gen_expr, opt_dict)

print(expr)