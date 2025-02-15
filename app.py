from fastapi import FastAPI, BackgroundTasks
from datasets import load_dataset
import httpx
import random
import tiktoken
import os

app = FastAPI()

DATASET = load_dataset("Laz4rz/wikipedia_science_chunked_small_rag_512")

API_URL = "https://ncs-client.condenses.ai"
API_KEY = os.environ.get("CONDENSE_API_KEY") 
HEADERS = {
    "user-api-key": API_KEY,
    "Content-Type": "application/json",
}


client = httpx.Client(base_url=API_URL, headers=HEADERS)


tokenizer = tiktoken.get_encoding("cl100k_base")


total_compression_results = []

def count_tokens(text: str) -> int:
    """Count the number of tokens in a given text."""
    return len(tokenizer.encode(text))

def compress_context():
    """
    Selects random contexts from the dataset, sends them to the compression API,
    and stores the results.
    """
    global total_compression_results
    results = []
    

    for _ in range(10):
        num_sample = random.randint(5, 10)  
        indices = random.sample(range(len(DATASET['train'])), num_sample)  
        selected_texts = [DATASET['train'][i]['text'] for i in indices]
        context = " ".join(selected_texts)  

        uncompressed_tokens = count_tokens(context)
        
        payload = {
            "context": context,
            "tier": "universal",
            "target_model": "llama",
            "miner_uid": -1,
            "top_incentive": 0.1,
        }

        try:
            # Send the request to the compression API
            response = client.post("/api/organic", json=payload, timeout=128)
            if response.status_code == 200:
                compressed_context = response.json().get("compressed_context", "")
                compressed_tokens = count_tokens(compressed_context)
                results.append({
                    "uncompressed": uncompressed_tokens,
                    "compressed": compressed_tokens,
                })
            else:
                # Indicate failure by setting compressed tokens to -1
                results.append({
                    "uncompressed": uncompressed_tokens,
                    "compressed": -1,
                })
        except Exception:
            # Handle request failures
            results.append({
                "uncompressed": uncompressed_tokens,
                "compressed": -1,
            })
    
    # Store results globally
    total_compression_results = results

@app.on_event("startup")
async def start_compression():
    """Run the background compression task when the server starts."""
    background_tasks = BackgroundTasks()
    background_tasks.add_task(compress_context)
    await background_tasks()

@app.get("/api/condenses-performance")
async def get_compression_results():
    """Endpoint to retrieve the latest compression performance results."""
    compress_context()  # Refresh compression results on each request
    return total_compression_results


