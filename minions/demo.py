import os
import time
from openai import AzureOpenAI
from mlx_lm.utils import load
from mlx_lm.generate import generate
from dotenv import load_dotenv
from huggingface_hub.utils.tqdm import disable_progress_bars

load_dotenv()

# suppress noisy hugging face stuff
disable_progress_bars()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

LOCAL_MODEL_PATH = "mlx-community/Phi-4-mini-instruct-8bit"
REMOTE_MODEL_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

QUANTUM_MECHANICS_HISTORY = """
The history of quantum mechanics is a fundamental part of the history of modern physics. The story began with Max Planck's 1900 quantum hypothesis that any energy-radiating atomic system can theoretically be divided into a number of discrete "energy elements" (quanta). This was a revolutionary idea that departed from classical physics. Planck devised this hypothesis to explain the observed frequency distribution of energy emitted by a black body.

Building on this, Albert Einstein in 1905 proposed that light itself is not a continuous wave, but consists of discrete quantum particles, which were later named photons. This theory helped explain the photoelectric effect, where shining light on certain materials can eject electrons. This was a pivotal moment, suggesting a dual wave-particle nature for light.

The next major leap came from Niels Bohr, who in 1913 proposed a new model for the atom. In the Bohr model, electrons travel in discrete, quantized orbits around the nucleus. He theorized that electrons could jump from one orbit to another, but could not exist in the space between orbits. This model successfully explained the spectral lines of the hydrogen atom.

Later, Louis de Broglie in 1924 put forward his hypothesis of matter waves, stating that all matter, not just light, exhibits wave-like properties. This concept was central to the development of wave mechanics. This was confirmed by experiments showing electron diffraction.

The full mathematical framework of quantum mechanics was developed in the mid-1920s. In 1925, Werner Heisenberg, along with Max Born and Pascual Jordan, developed matrix mechanics. Independently, Erwin Schrödinger developed wave mechanics and the non-relativistic Schrödinger equation in 1926, which described the evolution of a quantum system's wave function over time. Schrödinger subsequently showed that the two approaches were equivalent.
"""

LOCAL_MODEL_PROMPT_TEMPLATE = """You are a highly efficient data extraction assistant.
- Your task is to extract specific information from the provided text 'Context'.
- Provide ONLY the extracted data as your answer.
- Do NOT include explanations, apologies, or any conversational text.
- If the information is not found, respond with the single word: none.

Context:
---
{chunk}
---
Task: {job}
Answer:"""

# --- MODELS ---
def query_remote_lm(messages: list, temperature: float = 0.1):
    print("\n>>> Querying RemoteLM (Manager)...")
    client = AzureOpenAI(
        api_version="2024-02-01",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )
    start_time = time.time()
    response = client.chat.completions.create(
        model=REMOTE_MODEL_DEPLOYMENT,
        messages=messages,
        temperature=temperature,
        max_tokens=500
    )
    duration = time.time() - start_time
    content = response.choices[0].message.content
    print(f"<<< RemoteLM Response received in {duration:.2f}s")
    return content

def run_local_inference(model, tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        full_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        full_prompt = prompt

    response = generate(model, tokenizer, prompt=full_prompt, max_tokens=250, verbose=False)
    
    assistant_split_token = "<|assistant|>"
    if assistant_split_token in response:
        response = response.split(assistant_split_token)[-1]
    end_split_token = "<|end|>"
    if end_split_token in response:
        response = response.split(end_split_token)[0]

    return response.strip()


# --- MINIONS PROTOCOL IMPLEMENTATION ---
def prepare_jobs(user_query: str) -> list:
    print("\n--- Step 1: Job Preparation (RemoteLM) ---")
    print("RemoteLM is creating simple, targeted jobs based on the user query.")

    # For simplicity and determinism, we hardcode the jobs for this demo
    # normally it should be generated based on the user_query
    jobs = [
        "Find the contribution of Max Planck and the year it was made.",
        "Find the contribution of Albert Einstein and the year it was made.",
        "Find the contribution of Niels Bohr and the year it was made.",
        "Identify who developed matrix mechanics and in what year.",
        "Identify who developed wave mechanics and in what year."
    ]
    print(f"Jobs created: {jobs}")
    return jobs

def execute_jobs_locally(document: str, jobs: list) -> list:
    print("\n--- Step 2: Job Execution (LocalLM) ---")
    print(f"Loading local model from {LOCAL_MODEL_PATH}...")
    local_model, tokenizer = load(LOCAL_MODEL_PATH)

    chunk_size = 500
    chunks = [document[i:i+chunk_size] for i in range(0, len(document), chunk_size)]
    print(f"Document split into {len(chunks)} chunks.")

    job_results = []
    start_time = time.time()

    for i, chunk in enumerate(chunks):
        for job in jobs:
            prompt = LOCAL_MODEL_PROMPT_TEMPLATE.format(chunk=chunk, job=job)
            raw_result = run_local_inference(local_model, tokenizer, prompt)
            
            result_lower = raw_result.lower().strip()
            is_failure = any(phrase in result_lower for phrase in ["none", "not found", "not present", "cannot find", "not mentioned"])
            
            if not is_failure and raw_result:
                print(f"  - SUCCESS: Found relevant result in chunk {i+1}!")
                job_results.append(raw_result)

    duration = time.time() - start_time
    print(f"Local job execution finished in {duration:.2f}s.")
    print(f"Filtered results to be sent to RemoteLM: {job_results}")
    
    del local_model, tokenizer
    return job_results

def aggregate_and_synthesize(user_query: str, job_results: list) -> str:
    print("\n--- Step 3: Job Aggregation & Synthesis (RemoteLM) ---")
    
    system_prompt = """You are a science historian. You have received a list of facts extracted from a document about the history of physics.
- Your task is to synthesize this information into a clear, structured answer to the user's original query.
- Organize the information by scientist.
- Do not mention the extraction process, just provide the final answer."""

    results_str = "\n".join([f"- {res}" for res in job_results])
    
    user_prompt = f"Original Query: {user_query}\n\nExtracted Information:\n{results_str}\n\nPlease provide a final, synthesized answer."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    final_answer = query_remote_lm(messages)
    return final_answer

def main():
    start_time = time.time()
    print("="*50)
    print("      MINIONS Protocol Demo (Quantum Mechanics)")
    print("="*50)
    
    # user query
    user_query = "List the key contributions of Max Planck, Albert Einstein, and Niels Bohr to quantum mechanics, including the years of their discoveries."
    print(f"\nUser Query: {user_query}")
    
    jobs = prepare_jobs(user_query)
    results = execute_jobs_locally(QUANTUM_MECHANICS_HISTORY, jobs)
    
    if not results:
        print("\nLocalLM could not find any relevant information. Halting.")
        return
        
    final_answer = aggregate_and_synthesize(user_query, results)
    
    total_duration = time.time() - start_time
    
    print("\n--- FINAL ANSWER ---")
    print(final_answer)
    print("="*50)

    local_tokens = len(QUANTUM_MECHANICS_HISTORY.split())
    remote_tokens_sent = len(" ".join(results).split())
    print("\n--- Performance & Results Report ---")
    print(f"Total Workflow Duration: {total_duration:.2f}s")
    print("\nCost & Efficiency Analysis:")
    print(f"  - Tokens processed by FREE LocalLM: ~{local_tokens}")
    print(f"  - Tokens sent to EXPENSIVE RemoteLM API: ~{remote_tokens_sent}")
    if local_tokens > 0 and remote_tokens_sent > 0:
        token_reduction = (1 - (remote_tokens_sent / local_tokens)) * 100
        print(f"  - Token reduction for RemoteLM: {token_reduction:.1f}%")
    else:
        print("  - Token reduction for RemoteLM: 100.0%")
    print("\nAnswer Quality:")
    print("="*50)


if __name__ == "__main__":
    if not all([os.getenv("AZURE_OPENAI_ENDPOINT"), os.getenv("AZURE_OPENAI_API_KEY"), os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")]):
        print("ERROR: Please set the required Azure OpenAI environment variables.")
    else:
        main()