# Running LLMs locally

Simple case study for MADIVA LLM course

## Exercise 1: Use of vLLM

This should work easily on a Linux machine with a GPU or an Apple Silicon machine easily. If you are running
Linux without a GPU, more gymnastics are need as specialist packages must be installed. 

I am use using `pip3` but it is recommended to use frameworks such as `uv`. Python 3.12 is recommended but 3.10-3.13 shoudl work. 

```
pip3 install vllm torch
```

We are now ready to run the exercise. Read through the code `01-vllm-driver.py` and run.

```
python3 01-vllm-driver.py
```