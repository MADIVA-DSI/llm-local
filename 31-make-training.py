import json
import random
import csv


# plausible training examples for the study context



with open("data/metab_training.tsv", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    training_examples_raw = list(reader)

    
def format_example(ex):
    """Convert raw record into instruction-tuning format (Alpaca style)."""
    instruction = (
        "You are a metabolomics analyst supporting an RCT that compares "
        "dietary supplementation plus structured exercise versus control "
        "over 6 months. Outcomes of interest are cardio-metabolic health "
        "and systemic inflammation. Participants are adults in sub-Saharan "
        "Africa. Interpret the following metabolite in this context."
    )
    user_input = (
        f"What does a change in {ex['metabolite']} suggest in the context "
        f"of our exercise and dietary supplementation RCT?"
    )
    output = (
        f"Pathway: {ex['pathway']}\n\n"
        f"Direction and mechanism: {ex['direction']}\n\n"
        f"Study relevance: {ex['relevance']}\n\n"
        f"Caution: {ex['caution']}"
    )
    return {
        "instruction": instruction,
        "input":       user_input,
        "output":      output,
    }


dataset = [format_example(ex) for ex in training_examples_raw]

# ── Optional: augment with paraphrased inputs ────────────────────────────────
# Real fine-tuning benefits from input variation so the model generalises.
# Here we add simple paraphrases of each question.

paraphrase_templates = [
    "In our 6-month RCT, what is the significance of {} changing between baseline and endpoint?",
    "Our metabolomics collaborator measured {}. How should we interpret a significant change?",
    "Explain the role of {} as a biomarker for our study outcomes.",
    "A participant shows elevated {} at T1. What does this suggest?",
]

augmented = []
for ex in training_examples_raw:
    for template in paraphrase_templates:
        augmented.append({
            "instruction": format_example(ex)["instruction"],
            "input":       template.format(ex["metabolite"]),
            "output":      format_example(ex)["output"],
        })

dataset += augmented

random.shuffle(dataset)

print(f"Total training examples: {len(dataset)}")
# 20 originals + 20*4 paraphrases = 100 examples

with open("data/finetune_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)
