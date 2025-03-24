import json
import os
import time
import re
import openai
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional

# ========== Logging Configuration ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("kgqa_test.log"), logging.StreamHandler()]
)
logger = logging.getLogger("KG-Enhanced-QA")

# ========== API Configuration ==========
PANDALLA_API_KEY = ""
PANDALLA_API_BASE = ""

# ========== Initialize OpenAI Client ==========
client = openai.OpenAI(
    api_key=PANDALLA_API_KEY,
    base_url=PANDALLA_API_BASE
)

# ========== Knowledge Graph Configuration ==========
KG_JSON_PATH = r".\medical_graph_custom_format.json"

# Example: Load knowledge graph
def load_knowledge_graph(kg_path: str) -> Dict[str, Any]:
    if not os.path.exists(kg_path):
        logger.warning(f"Knowledge graph file does not exist: {kg_path}")
        return {}
    with open(kg_path, "r", encoding="utf-8") as f:
        graph_data = json.load(f)
    return graph_data

# Simple example: Retrieve or process information in the knowledge graph based on user questions
def query_knowledge_graph(user_question: str, kg_data: Dict[str, Any]) -> str:
    """
    Write this based on business requirements:
    1. Perform simple retrieval or keyword matching in kg_data
    2. Return the found text content, or an empty string
    """
    # Example: Simple keyword matching
    matched_info = []
    for node_key, node_val in kg_data.items():
        # Write more complex retrieval logic as needed
        if node_key in user_question:
            matched_info.append(f"【{node_key}】Related information: {node_val}")
    if matched_info:
        return "\n".join(matched_info)
    return ""

def build_prompt(user_question: str, knowledge_info: str) -> str:
    """
    Assemble the retrieved information from the knowledge graph and user question into a conversation prompt.
    Can design System / User / Context more precisely.
    """
    system_msg = (
        "You are a professional medical assistant. "
        "If the knowledge graph information below can answer the user's question, please cite it appropriately; "
        "if it's not relevant, please answer based on your existing knowledge."
    )
    user_msg = f"User question: {user_question}"
    if knowledge_info:
        user_msg += f"\n\n【Knowledge Graph Supplementary Information】:\n{knowledge_info}"
    prompt = f"{system_msg}\n\n{user_msg}"
    return prompt

def ask_model(user_question: str, model_name: str = "llama-3.1-405b") -> str:
    """
    Core function: Use Pandalla's API to call the specified model (e.g., llama-3.1-405b / deepseek-r1) to complete the answer
    """
    # 1. Load knowledge graph (if large, can be loaded once globally)
    kg_data = load_knowledge_graph(KG_JSON_PATH)
    # 2. Query knowledge graph
    knowledge_info = query_knowledge_graph(user_question, kg_data)
    # 3. Construct conversation prompt
    prompt_text = build_prompt(user_question, knowledge_info)
    
    # 4. Call Pandalla's ChatCompletion
    #    Note: Under OpenAI compatible SDK, conversations need to be passed in as a messages array
    #    Here is a simplified example
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user",   "content": prompt_text}
    ]
    
    logger.info(f"Calling model: {model_name}, Prompt length: {len(prompt_text)}")
    
    response = client.chat.completions.create(
        model=model_name,  # For example "llama-3.1-405b" or "deepseek-r1" or "claude-3-5-sonnet-latest" or "gemini-1.5-pro"
        messages=messages,
        max_tokens=800,     # Can be adjusted according to requirements
        temperature=0.7,    # Can be adjusted according to requirements
        top_p=0.9,          # Can be adjusted according to requirements
        n=1,                # Number of answers to return
        stream=False        # Whether to return as a stream
    )
    
    # Extract text according to the interface return format
    # For OpenAI compatible API, typically response["choices"][0]["message"]["content"]
    if "choices" in response and len(response["choices"]) > 0:
        return response["choices"][0]["message"]["content"]
    else:
        return "Sorry, I cannot provide an answer."

if __name__ == "__main__":
    # Test example
    user_question = "I've been having a cold and cough recently, what treatment should I seek?"
    # You can change the model name here anytime
    # deepseek-r1 or llama-3.1-405b, etc.
    answer = ask_model(user_question, model_name="deepseek-r1")
    print("=== Model Answer ===")
    print(answer)