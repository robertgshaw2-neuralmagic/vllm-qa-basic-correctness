from openai import OpenAI
from datasets import load_dataset
from transformers import AutoTokenizer
import argparse
import time
from threading import Thread

parser = argparse.ArgumentParser()
parser.add_argument('--num-requests', type=int, default=15, help="Number of total requests.")
parser.add_argument('--request-rate', type=int, default=1.0, help="Request per second.")
parser.add_argument('--num-turns', type=int, default=1, help="Number of requests.")

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
ds = load_dataset("nm-testing/qa-chat-prompts", split="train_sft")

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

NUM_TURNS = 1
def run(idx, messages, num_turns=1):
    print(f"STARTING THREAD_IDX: {idx}")

    chat_completion = client.chat.completions.create(
        messages=messages[:num_turns],
        model=model,
        max_tokens=200,
    )

    print("------------------------------------------")
    print(f"FINISHED THREAD_IDX: {idx}")
    print("------------------------------------------")
    print(f"Prompt:\n{tokenizer.apply_chat_template(messages[:num_turns], tokenize=False)}")
    print("------------------------------------------")
    print(f"Response:\n{chat_completion.choices[0].message.content}")
    print("------------------------------------------")
    print("\n\n\n")

if __name__ == "__main__":
    args = parser.parse_args()

    if args.num_requests > len(ds):
        print(f"MAX OF {len(ds)} requests allowed.")
    else:
        ds = ds.select(range(args.num_requests))
        
    ts = [Thread(target=run, args=[idx, messages, args.num_turns]) for idx, messages in enumerate(ds["messages"])]
    
    for t in ts:
        t.start()
        time.sleep(args.request_rate)
    for t in ts:
        t.join()