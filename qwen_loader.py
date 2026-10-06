from transformers import AutoProcessor, AutoModelForCausalLM

model_name = "Qwen/Qwen3.5-0.8B"
custom_path = "./Qwen3.5-0.8B/"

processor = AutoProcessor.from_pretrained(model_name, cache_dir=custom_path)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=custom_path,
    device_map="auto",
    torch_dtype="auto",
)