

from vllm import LLM, SamplingParams
import torch

prompts = ["Who won the 1972 FA Cup?",
           "What are the pros and cons of doing Base Quality Score Recalibration in variant calling?",
           "What is the best treatment for shin splits caused by a running injury"]
sampling_params = SamplingParams(temperature=0.0, max_tokens=512)



llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.3")
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)

del llm
torch.cuda.empty_cache()
