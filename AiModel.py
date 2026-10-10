
from transformers import pipeline

model_id = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading model...")

generator = pipeline(
    "text-generation",
    model=model_id,
    device="cpu",
)

print("Model loaded successfully!")

while True:
    question = input("1\nYou: ")

    if question.lower() in ["exit", "quit"]:
        break

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "Answer questions clearly and briefly. "
                "For simple arithmetic, calculate carefully."
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    result = generator(
        messages,
        max_new_tokens=512,
        do_sample=False,
        return_full_text=False,
    )

    generated = result[0]["generated_text"]

    if isinstance(generated, str):
        answer = generated
    elif isinstance(generated, list):
        answer = generated[-1].get("content", str(generated[-1]))
    else:
        answer = str(generated)

    print("AI:", answer)