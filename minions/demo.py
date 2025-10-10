import os
import time
import json
from openai import AzureOpenAI
from mlx_lm.utils import load
from mlx_lm.generate import generate
from dotenv import load_dotenv
from huggingface_hub.utils.tqdm import disable_progress_bars

load_dotenv("../.env")

# suppress noisy hugging face stuff
disable_progress_bars()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

LOCAL_MODEL_PATH = "mlx-community/Phi-4-mini-instruct-8bit"

def require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val

AOI_MODEL_DEPLOYMENT = require_env("AZURE_OPENAI_DEPLOYMENT_NAME")
AOI_RESOURCE = require_env("AZURE_OPENAI_RESOURCE")
AOI_KEY = require_env("AZURE_OPENAI_KEY")

with open("quantum_mechanics_history.txt", "r", encoding="utf-8") as f:
    QUANTUM_MECHANICS_HISTORY = f.read()

LOCAL_MODEL_PROMPT_TEMPLATE_ROBUST = """You are a meticulous and literal fact-checker. Your process is a strict two-step evaluation:
1.  First, analyze the 'Context' to determine if it contains any information that can directly answer the 'Task'.
2.  If the 'Context' is NOT relevant to the 'Task', you MUST immediately stop and respond with the single word: none.

- If and ONLY IF the information is present, extract the relevant facts verbatim or as a close paraphrase.
- Do NOT invent, guess, or mix information from different people or concepts. If the primary subject of the 'Task' (e.g., a person's name) is not mentioned in the 'Context', the answer is always 'none'.
- Your final output must be ONLY the extracted data or the word 'none'.

Context:
---
{chunk}
---
Task: Based ONLY on the text in the 'Context' above, {job}
"""

# --- MODELS ---
def query_remote_lm(messages: list, temperature: float = 0.1):
    print("\n>>> Querying RemoteLM (Manager)...")
    client = AzureOpenAI(
        api_version="2024-06-01",
        azure_endpoint=f"https://{AOI_RESOURCE}.openai.azure.com",
        api_key=AOI_KEY,
    )
    start_time = time.time()
    response = client.chat.completions.create(
        model=AOI_MODEL_DEPLOYMENT,
        messages=messages,
        temperature=temperature,
        max_tokens=500,
    )
    duration = time.time() - start_time
    content = response.choices[0].message.content or ""
    print(f"<<< RemoteLM Response received in {duration:.2f}s")
    return content


def run_local_inference(model, tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]

    if (
        hasattr(tokenizer, "apply_chat_template")
        and tokenizer.chat_template is not None
    ):
        full_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        full_prompt = prompt

    response = generate(
        model, tokenizer, prompt=full_prompt, max_tokens=250, verbose=False
    )

    assistant_split_token = "<|assistant|>"
    if assistant_split_token in response:
        response = response.split(assistant_split_token)[-1]
    end_split_token = "<|end|>"
    if end_split_token in response:
        response = response.split(end_split_token)[0]

    return response.strip()


# --- MINIONS PROTOCOL IMPLEMENTATION ---
def prepare_jobs(user_query: str, use_predefined_jobs: bool = False) -> tuple[list, int]:
    print("\n--- Step 1: Job Preparation (RemoteLM) ---")
    print("RemoteLM is creating simple, targeted jobs based on the user query.")

    if not use_predefined_jobs:
        system_prompt = """You are a task decomposition expert. Your job is to break down a user's complex query into simple, atomic extraction tasks.

    Rules:
    - Each task should be a single, focused question that can be answered from a text chunk
    - Tasks should be specific and actionable (e.g., "Find X and the year Y") and intended for finding information in the attached text
    - Create 3-7 tasks depending on the complexity of the query
    - Return ONLY a JSON array of task strings, nothing else
    - Format: ["task 1", "task 2", "task 3"]"""

        user_prompt = f"""User Query: {user_query}

    Break this query down into simple extraction tasks. Return only the JSON array."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = query_remote_lm(messages, temperature=0.3)

        try:
            response = response.strip()
            if not response.startswith("["):
                start_idx = response.find("[")
                end_idx = response.rfind("]") + 1
                if start_idx != -1 and end_idx > start_idx:
                    response = response[start_idx:end_idx]

            jobs = json.loads(response)
            print(f"Jobs created: {jobs}")
            return jobs, len(system_prompt) + len(user_prompt)
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse JSON response. Error: {e}")
            print(f"Raw response: {response}")
            print("Falling back to predefined jobs.")

    jobs = [
        "Find the contribution of Max Planck and the year it was made.",
        "Find the contribution of Albert Einstein and the year it was made.",
        "Find the contribution of Niels Bohr and the year it was made."
    ]
    print(f"Using fallback jobs: {jobs}")
    return jobs, 0


def execute_jobs_locally(document: str, jobs: list) -> tuple[list, int]:
    print("\n--- Step 2: Job Execution (LocalLM) ---")
    print(f"Loading local model from {LOCAL_MODEL_PATH}...")
    local_model, tokenizer = load(LOCAL_MODEL_PATH)

    chunk_size = 500
    chunks = [document[i : i + chunk_size] for i in range(0, len(document), chunk_size)]
    print(f"Document split into {len(chunks)} chunks.")

    job_results = []
    start_time = time.time()
    total_chars_processed = 0

    for i, chunk in enumerate(chunks):
        for job in jobs:
            prompt = LOCAL_MODEL_PROMPT_TEMPLATE_ROBUST.format(chunk=chunk, job=job)
            total_chars_processed += len(prompt)
            raw_result = run_local_inference(local_model, tokenizer, prompt)

            result_lower = raw_result.lower().strip()
            is_failure = result_lower == "none"

            if not is_failure and raw_result:
                print(f"  - SUCCESS: Found relevant result in chunk {i+1}!")
                job_results.append(raw_result)
            else:
                print(f"  - No relevant info found in chunk {i+1}.")

            print(f"    (LocalLM response: '{raw_result}')")

    duration = time.time() - start_time
    print(f"Local job execution finished in {duration:.2f}s.")
    print(f"Filtered results to be sent to RemoteLM: {job_results}")

    del local_model, tokenizer
    return job_results, total_chars_processed


def aggregate_and_synthesize(user_query: str, job_results: list) -> tuple[str, int]:
    print("\n--- Step 3: Job Aggregation & Synthesis (RemoteLM) ---")

    system_prompt = """You are a science historian. You have received a list of facts extracted from a document about the history of physics.
- Your task is to synthesize this information into a clear, structured answer to the user's original query.
- Organize the information by scientist.
- Do not mention the extraction process, just provide the final answer."""

    results_str = "\n".join([f"- {res}" for res in job_results])

    user_prompt = f"Original Query: {user_query}\n\nExtracted Information:\n{results_str}\n\nPlease provide a final, synthesized answer."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    final_answer = query_remote_lm(messages)
    return final_answer, len(system_prompt) + len(user_prompt)


def evaluate_response_quality(
    user_query: str, document: str, final_answer: str
) -> tuple[int, str]:
    print("\n--- Step 4: Response Quality Evaluation (RemoteLM) ---")
    print("RemoteLM is evaluating the quality of the generated answer...")

    system_prompt = """You are an expert evaluator of AI-generated responses. Your task is to assess the quality of an answer given the original document, user query, and the generated response.

Please evaluate the response on a scale of 1-5 where:
1 = Very Poor (completely inaccurate or irrelevant)
2 = Poor (mostly inaccurate with some relevant information)
3 = Fair (some accuracy but missing key information or has notable errors)
4 = Good (mostly accurate and complete with minor issues)
5 = Excellent (highly accurate, complete, and well-structured)

Consider these criteria:
- Accuracy: Does the answer correctly reflect the information in the source document?
- Completeness: Does it address all parts of the user's query?
- Clarity: Is the answer well-organized and easy to understand?
- Relevance: Does it stay focused on what was asked?

Provide your evaluation in this exact format:
Score: [1-5]
Reasoning: [Brief explanation of your assessment]"""

    user_prompt = f"""Original Document:
{document}

User Query: {user_query}

Generated Answer:
{final_answer}

Please evaluate this response."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    evaluation_response = query_remote_lm(messages, temperature=0.3)

    # Handle case where evaluation_response might be None
    if evaluation_response is None:
        return 0, "Error: No evaluation response received"

    # Extract score from response
    try:
        score_line = [
            line
            for line in evaluation_response.split("\n")
            if line.startswith("Score:")
        ][0]
        score = int(score_line.split(":")[1].strip())
    except (IndexError, ValueError):
        score = 0  # Default if parsing fails

    return score, evaluation_response


def main():
    start_time = time.time()
    print("=" * 50)
    print("      MINIONS Protocol Demo (Quantum Mechanics)")
    print("=" * 50)

    # user query
    user_query = "what did Planck, Einstein, and Bohr contribute to quantum mechanics?"
    print(f"\nUser Query: {user_query}")

    jobs, prep_job_chars = prepare_jobs(user_query)
    results, local_chars_processed = execute_jobs_locally(QUANTUM_MECHANICS_HISTORY, jobs)

    if not results:
        print("\nLocalLM could not find any relevant information. Halting.")
        return

    final_answer, aggregate_chars = aggregate_and_synthesize(user_query, results)

    # Evaluate the response quality
    evaluation_score, evaluation_details = evaluate_response_quality(
        user_query, QUANTUM_MECHANICS_HISTORY, final_answer
    )

    total_duration = time.time() - start_time

    print("\n--- FINAL ANSWER ---")
    print(final_answer)
    print("=" * 50)

    print("\n--- RESPONSE QUALITY EVALUATION ---")
    print(f"Quality Score: {evaluation_score}/5")
    print(f"Evaluation Details:\n{evaluation_details}")
    print("=" * 50)

    remote_chars_sent = prep_job_chars + aggregate_chars

    print("\n--- Performance & Results Report ---")
    print(f"Total Workflow Duration: {total_duration:.2f}s")
    print("\nCost & Efficiency Analysis (using character counts):")
    print(f"  - Characters processed by FREE LocalLM: ~{local_chars_processed}")
    print(f"  - Characters sent to EXPENSIVE RemoteLM API (Job Prep + Synthesis): ~{remote_chars_sent}")

    print("\nAnswer Quality:")
    print(f"  - AI Judge Score: {evaluation_score}/5")
    print("=" * 50)


if __name__ == "__main__":
    main()
