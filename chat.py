from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained(
    "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
)

prompt = "She ask: just answer the name of the capital of India and we will stop talking. He answer:"

for word in llm(prompt, stream=True):
    print(word, end="", flush=True)
print()
