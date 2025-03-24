from tqdm import tqdm
import json
from config import logger

def run_baseline_evaluation(self, test_data, output_file="baseline_medqa_results.json", sample_limit=None):
    # Preprocess data to ensure consistent format
    processed_data = self.preprocess_data(test_data)
    
    results = []
    correct_count = 0
    
    # Optionally limit the number of test samples
    if sample_limit and sample_limit < len(processed_data):
        processed_data = processed_data[:sample_limit]
        logger.info(f"Limited to {sample_limit} samples.")
    
    for i, sample in enumerate(tqdm(processed_data, desc="Baseline")):
        question = sample["question"]
        options = sample["options"]
        
        # Determine the correct answer
        correct_answer = None
        if sample.get("answer_idx") is not None:
            # If answer_idx is provided and valid
            answer_idx = sample.get("answer_idx")
            if isinstance(answer_idx, int) and 0 <= answer_idx < len(options):
                correct_answer = chr(65 + answer_idx)
        
        if not correct_answer and isinstance(sample.get("answer"), str):
            answer_text = sample.get("answer", "")
            for char in answer_text.upper():
                if char in "ABCD":
                    correct_answer = char
                    break
        
        if not correct_answer:
            correct_answer = "A"
        
        try:
            formatted_question = f"""
            Question:
            {question}
            
            Options:
            A. {options[0]}
            B. {options[1]}
            C. {options[2]}
            D. {options[3]}
            
            Please select the most accurate answer.
            """
            
            logger.info(f"Questions {i+1}/{len(processed_data)}")
            
            # Direct use of LLM
            system_prompt = """You are a professional medical expert taking a multiple-choice medical exam.
            Choose the most accurate answer based on your medical knowledge.
            Only respond with the option letter (A, B, C, or D)."""
            
            response = self.llm_interface.ask(
                formatted_question,
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=50
            )
            
            # Extract predicted answer
            predicted_answer = None
            for char in response.strip().upper():
                if char in "ABCD":
                    predicted_answer = char
                    break
            
            if not predicted_answer:
                predicted_answer = "A"
            
            # Check if prediction is correct
            is_correct = (predicted_answer == correct_answer)
            if is_correct:
                correct_count += 1
            
            # Print results
            result_str = f"Baseline: {predicted_answer}" + (" ✓ Correct" if is_correct else f" ✗ Wrong (Correct answer: {correct_answer})")
            logger.info(result_str)
            
            # Calculate current accuracy
            current_accuracy = correct_count / (i + 1)
            logger.info(f"{correct_count}/{i+1} = {current_accuracy:.2%}")
            
            # Save results
            result = {
                "id": i,
                "question": question,
                "options": options,
                "correct_answer": correct_answer,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "raw_response": response
            }
            
        except Exception as e:
            logger.error(str(e))
            result = {
                "id": i,
                "question": question,
                "options": options if isinstance(options, list) else [],
                "correct_answer": correct_answer,
                "predicted_answer": "Error",
                "is_correct": False,
                "error": str(e)
            }
        
        results.append(result)
        
        # Save intermediate results
        if (i + 1) % 5 == 0 or (i + 1) == len(processed_data):
            accuracy_so_far = correct_count / (i + 1)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "results": results,
                    "accuracy": accuracy_so_far,
                    "processed": i + 1,
                    "total": len(processed_data)
                }, f, indent=2, ensure_ascii=False)
    
    # Calculate final accuracy
    accuracy = correct_count / len(processed_data) if processed_data else 0
    
    # Save final results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "results": results,
            "accuracy": accuracy,
            "processed": len(processed_data),
            "total": len(processed_data)
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Baseline final accuracy: {accuracy:.2%}")
    return accuracy, results