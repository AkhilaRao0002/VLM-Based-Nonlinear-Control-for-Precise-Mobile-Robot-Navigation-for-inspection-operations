import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForMultimodalLM


MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"

IMAGE = r"..\tartanair_data\AbandonedFactory\Data_omni\P0000\image_lcam_front\000010_lcam_front.png"


print("[TEST] Loading image...")
image = Image.open(IMAGE).convert("RGB")


print("[TEST] Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL)


print("[TEST] Loading model...")
model = AutoModelForMultimodalLM.from_pretrained(MODEL)


device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)
model.eval()

print("[TEST] Device:", device)

print("[TEST] GPU:", torch.cuda.get_device_name(0)
      if torch.cuda.is_available()
      else "CPU")


# ============================================================
# ROBOT INSPECTION PROMPT
# ============================================================

prompt = """
Describe this image in detail.

Identify:
- objects
- furniture
- structures
- vehicles
- people
- walls
- floors
- ceilings

Write a short description of everything you can actually see.
Do not guess.
"""


messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image"
            },
            {
                "type": "text",
                "text": """
Analyze this image for a mobile robot navigating an industrial environment.

Answer these questions:

1. What type of environment is visible?
2. What obstacles are clearly visible?
3. What type of ground or floor is visible?
4. Is there visible open space where a robot could potentially move?
5. Is the path blocked or partially blocked?
6. Are there any people or vehicles?
7. Give one short navigation observation.

Do not guess. Only describe what is visible.
"""
            }
        ]
    }
]

# ============================================================
# CHAT TEMPLATE
# ============================================================

chat_text = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False
)


# ============================================================
# PROCESS IMAGE + TEXT
# ============================================================

inputs = processor(
    text=chat_text,
    images=image,
    return_tensors="pt"
)


inputs = {
    key: value.to(device)
    if hasattr(value, "to")
    else value
    for key, value in inputs.items()
}


# ============================================================
# GENERATION
# ============================================================

print("[TEST] Generating...")


with torch.no_grad():

    generated_ids = model.generate(
    **inputs,
    max_new_tokens=150,
    do_sample=False
)


input_length = inputs["input_ids"].shape[-1]

generated_ids = generated_ids[:, input_length:]


response = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True
)[0].strip()


# ============================================================
# RESULT
# ============================================================

print()
print("=" * 60)
print("VLM INDUSTRIAL INSPECTION TEST")
print("=" * 60)

print(response)

print("=" * 60)