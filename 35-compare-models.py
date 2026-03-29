
from unsloth import FastLanguageModel



PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""


model = "unsloth/Llama-3.2-3B-Instruct"
# Swap for llama3:8b or mistral:7b if you want stronger baseline quality.

# Step 1 — load the base model again (same as in training)
original_model, tokeniser = FastLanguageModel.from_pretrained(
    model_name     = model,
    max_seq_length = 1024,
    load_in_4bit   = True,
)
FastLanguageModel.for_inference(original_model)


# Step 2: Add the adapters
ft_model, _ = FastLanguageModel.from_pretrained(
    model_name     = "data/metabolomics_lora",  # adapter path directly
    max_seq_length = 1024,
    load_in_4bit   = True,
)
FastLanguageModel.for_inference(ft_model)



def query_model(metabolite, model, tokeniser):
    prompt = PROMPT_TEMPLATE.format(
        instruction = (
            "You are a metabolomics analyst supporting an RCT that compares "
            "dietary supplementation plus structured exercise versus control "
            "over 6 months. Outcomes of interest are cardio-metabolic health "
            "and systemic inflammation. Participants are adults in sub-Saharan "
            "Africa. Interpret the following metabolite in this context."
        ),
        input  = f"What does a change in {metabolite} suggest in the context of our RCT?",
        output = "",   # leave blank -- model completes this
    )
    inputs = tokeniser(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens    = 300,
        temperature       = 0.7,     
        do_sample         = True,    
        repetition_penalty = 1.3,    
        eos_token_id      = tokeniser.eos_token_id,
    )
    decoded = tokeniser.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("### Response:")[-1].strip()

print("=== BASE MODEL ===")

print(query_model("TMAO", original_model, tokeniser))

print("\n=== FINE-TUNED MODEL ===")
print(query_model("TMAO", ft_model, tokeniser))
