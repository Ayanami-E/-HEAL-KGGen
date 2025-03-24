import json
import re
from config import logger, client

class LLMInterface:
    """Interface for interacting with large language models"""
    
    def __init__(self, model="grok-2-1212", client=None):#Choose from here to use the model
        """Initialize LLM interface"""
        self.model = model
        self.client = client
        self.cache = {}  # For caching responses
        logger.info(f"LLM interface initialized, using model: {model}")
    
    def ask(self, prompt, system_prompt=None, temperature=0.3, max_tokens=1500):
        """Ask a question to the language model"""
        # Format prompt to ensure we get a clear letter answer
        formatted_prompt = f"""
{prompt}

IMPORTANT: Your answer must be a single letter (A, B, C, or D) corresponding to the correct option.
Analyze all evidence carefully, focusing on the knowledge graph relationships.
Just respond with the letter of the answer, nothing else.
"""
        
        # Create cache key
        cache_key = f"{formatted_prompt}_{system_prompt}_{temperature}_{max_tokens}_{self.model}"
        
        # Check cache
        if cache_key in self.cache:
            logger.info("Using cached LLM response")
            return self.cache[cache_key]
        
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": formatted_prompt})
        
        # Add a few-shot examples to help the model understand the task better
        messages.append({"role": "assistant", "content": "A"})
        messages.append({"role": "user", "content": "Was this correct? Can you verify your answer by checking the knowledge graph relationships?"})
        messages.append({"role": "assistant", "content": "B"})
        
        try:
            logger.info(f"Calling LLM API, prompt length: {len(formatted_prompt)}")
            
            if self.client is None:
                logger.warning("No LLM client provided, returning simulated response")
                return "A"  # Simulated response, for testing only
            
            # Add retry mechanism
            max_retries = 3
            for retry in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    
                    result = response.choices[0].message.content
                    
                    # Verify we have a single letter answer
                    letter_answer = self.get_single_letter_answer(result)
                    if letter_answer:
                        # Cache result
                        self.cache[cache_key] = result
                        return result
                    elif retry < max_retries - 1:
                        # If not a clear letter and we have retries left, ask again with stronger instruction
                        messages.append({"role": "assistant", "content": result})
                        messages.append({"role": "user", "content": "Please provide just a single letter (A, B, C, or D) as your answer."})
                        continue
                    else:
                        # Last retry, just return what we got
                        self.cache[cache_key] = result
                        return result
                        
                except Exception as inner_e:
                    if retry < max_retries - 1:
                        logger.warning(f"Retry {retry+1}/{max_retries} after error: {inner_e}")
                        continue
                    else:
                        raise inner_e
            
        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            return f"Error: Could not get response - {str(e)}"
    
    def extract_entities(self, text, entity_types=None):
        """Extract entities from text"""
        if entity_types is None:
            entity_types = ["disease", "gene", "symptom", "treatment", "biomarker", "drug", "protein"]
        
        prompt = f"""
Extract and classify entities from the following text:

Text: {text}
Entity types to extract: {', '.join(entity_types)}

Return results in JSON format, with a list of entities for each category.
"""
        
        system_prompt = """You are an assistant specialized in extracting entities from medical text. Only return the entity list in JSON format, without any additional explanation or comments."""
        
        response = self.ask(prompt, system_prompt, temperature=0.1)
        
        # Try to parse JSON response
        try:
            # Look for JSON part in response
            json_text = response
            if "```json" in response:
                json_text = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_text = response.split("```")[1].split("```")[0]
            
            entities = json.loads(json_text)
            return entities
        except Exception as e:
            logger.error(f"Error parsing entity extraction response: {e}")
            logger.debug(f"Original response: {response}")
            return {category: [] for category in entity_types}
    
    def get_single_letter_answer(self, response):
        """Extract a single letter answer (A, B, C, or D) from the response"""
        if not response:
            logger.warning("Empty response received")
            return None
        
        # Clean the response
        clean_response = response.strip().upper()
        
        # Pattern 1: Direct single letter
        if clean_response in "ABCD":
            return clean_response
        
        # Pattern 2: Letter followed by period or parenthesis (A. or A))
        match = re.search(r"\b([A-D])[\.\)]", clean_response)
        if match:
            return match.group(1)
        
        # Pattern 3: Answer: X or The answer is X
        match = re.search(r"ANSWER\s*:?\s*([A-D])", clean_response)
        if match:
            return match.group(1)
        
        match = re.search(r"THE\s+ANSWER\s+IS\s+([A-D])", clean_response)
        if match:
            return match.group(1)
        
        # Pattern 4: Option X is correct
        match = re.search(r"OPTION\s+([A-D])", clean_response)
        if match:
            return match.group(1)
        
        # Pattern 5: Find first occurrence of A, B, C, D
        for char in clean_response:
            if char in "ABCD":
                return char
        
        logger.warning(f"Could not extract valid answer letter from: {response[:100]}...")
        return None
    
    def chain_of_thought_reasoning(self, question, options, kg_data=None):
        """Apply chain-of-thought reasoning to medical questions"""
        formatted_options = "\n".join([f"{key}: {value}" for key, value in options.items()])
        
        kg_context = ""
        if kg_data:
            kg_context = f"""
Knowledge Graph Information:
{kg_data}
            """
        
        prompt = f"""
I need you to solve this medical question through careful reasoning:

Question: {question}

Options:
{formatted_options}

{kg_context}

Please think step by step:
1. First, identify key medical entities in the question and options
2. Analyze the relationships between these entities based on the knowledge graph
3. Compare each option against these relationships
4. Determine which option is most consistent with the evidence

Your final answer should be a single letter (A, B, C, or D).
"""
        
        system_prompt = """You are a medical expert solving complex medical exam questions. 
Use careful reasoning to analyze the question and determine the most accurate answer."""
        
        response = self.ask(prompt, system_prompt, temperature=0.2, max_tokens=2000)
        
        # Extract the letter answer
        answer_letter = self.get_single_letter_answer(response)
        
        return answer_letter or "A", response