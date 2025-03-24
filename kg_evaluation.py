from tqdm import tqdm
import json
from config import logger

def _get_kg_enhanced_prediction(self, question, options, sample):
    """Use knowledge graph enhanced answer prediction"""
    logger.info("KG Enhanced start predicting")
    
    # Convert options list to dictionary format
    options_dict = {
        "A": options[0],
        "B": options[1],
        "C": options[2],
        "D": options[3]
    }
    
    # Use the new KG enhancement approach
    from kg_enhancement import ask_kg_enhanced_question
    
    answer_letter, enhanced_prompt = ask_kg_enhanced_question(
        question=question,
        options=options_dict,
        kg_handler=self.kg_handler,
        llm_interface=self.llm_interface
    )
    
    logger.info(f"KG enhanced result: {answer_letter}")
    return answer_letter

def run_evaluation(self, test_data, output_file="kg_enhanced_medqa_results.json", sample_limit=None):
    """Evaluate medical QA data using KG enhancement"""
    # Preprocess data
    processed_data = self.preprocess_data(test_data)
    
    # Optionally limit the number of test samples
    if sample_limit and sample_limit < len(processed_data):
        processed_data = processed_data[:sample_limit]
        logger.info(f"Limited to {sample_limit} samples")
    
    # Initialize results
    results = []
    correct_count = 0
    
    # Process each sample
    for i, sample in enumerate(tqdm(processed_data, desc="Processing")):
        try:
            logger.info(f"Question {i+1}/{len(processed_data)}")
            
            # Extract question components
            question = sample["question"]
            options = sample["options"]
            
            # Determine correct answer
            correct_answer = None
            if sample.get("answer_idx") is not None:
                # If answer_idx is provided and valid
                answer_idx = sample.get("answer_idx")
                if isinstance(answer_idx, int) and 0 <= answer_idx < len(options):
                    correct_answer = chr(65 + answer_idx)  # Convert to A, B, C, D
            
            if not correct_answer and isinstance(sample.get("answer"), str):
                # Try to extract letter from answer string
                answer_text = sample.get("answer", "")
                for char in answer_text.upper():
                    if char in "ABCD":
                        correct_answer = char
                        break
            
            if not correct_answer:
                # If no valid answer found, default to A
                correct_answer = "A"
            
            # Extract medical entities
            entities = self.extract_medical_entities(question)
            entities_info = [{"id": e["id"], "name": e["name"], "category": e["category"]} for e in entities]
            
            # Retrieve knowledge subgraph
            subgraph_info = self.retrieve_knowledge_subgraph(entities)
            has_knowledge = len(subgraph_info["entities"]) > 0 or len(subgraph_info["relationships"]) > 0
            knowledge_size = len(subgraph_info["entities"]) + len(subgraph_info["relationships"])
            
            # Use KG enhanced method to get prediction
            predicted_answer = self._get_kg_enhanced_prediction(question, options, sample)
            
            # Determine if prediction is correct
            is_correct = (predicted_answer == correct_answer)
            if is_correct:
                correct_count += 1
            
            # Print results
            result_str = f"Prediction: {predicted_answer}" + (" ✓ Correct" if is_correct else f" ✗ Wrong (Correct answer: {correct_answer})")
            print(result_str)
            
            # Calculate current accuracy
            accuracy = correct_count / (i + 1)
            print(f"Current accuracy: {correct_count}/{i+1} = {accuracy:.2%}")
            
            # Save result with additional information
            result = {
                "id": i,
                "question": question,
                "options": options,
                "correct_answer": correct_answer,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "entities_extracted": entities_info,
                "knowledge_used": has_knowledge,
                "knowledge_size": knowledge_size
            }
            results.append(result)
            
            # Save checkpoint results
            if (i + 1) % 5 == 0 or i == len(processed_data) - 1:
                logger.info(f"{i+1}/{len(processed_data)} questions processed. Current accuracy: {accuracy:.2%}")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "results": results,
                        "accuracy": accuracy,
                        "processed": i + 1,
                        "total": len(processed_data)
                    }, f, indent=2, ensure_ascii=False)
                    
        except Exception as e:
            logger.error(f"{e}")
            logger.error(str(e))
            # Add failed result
            result = {
                "id": i,
                "question": sample.get("question", ""),
                "error": str(e),
                "is_correct": False
            }
            results.append(result)
    
    # Calculate final accuracy
    final_accuracy = correct_count / len(processed_data) if processed_data else 0
    
    # Save final results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "results": results,
            "accuracy": final_accuracy,
            "processed": len(processed_data),
            "total": len(processed_data)
        }, f, indent=2, ensure_ascii=False)
    
    # Generate detailed report
    self.generate_detailed_report(results)
    
    logger.info(f"Final accuracy: {final_accuracy:.2%}")
    return final_accuracy, results