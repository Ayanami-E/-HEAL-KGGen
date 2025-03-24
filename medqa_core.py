import json
import os
from tqdm import tqdm
import re
from config import logger
from kg_handler import JSONKnowledgeGraphHandler
from llm_interface import LLMInterface

class MedQATestFramework:
    """Framework for medical question answering testing using knowledge graph enhanced multi-agent system"""
    
    def __init__(self, kg_handler=None, llm_interface=None):
        """Initialize test framework"""
        # Initialize knowledge graph handler
        self.kg_handler = kg_handler
        
        # Initialize LLM interface
        self.llm_interface = llm_interface or LLMInterface(model="gpt-4")
        
        logger.info("Initialized")
    
    def load_medqa_data(self, file_path):
        """Load medical QA data from JSONL file"""
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        sample = json.loads(line.strip())
                        data.append(sample)
                    except json.JSONDecodeError:
                        logger.warning(f"{line}")
        except Exception as e:
            logger.error(f"{e}")
        return data
    
    def preprocess_data(self, data):
        """Preprocess medical QA data for evaluation"""
        processed_data = []
        
        for sample in data:
            try:
                # Process question
                question = sample.get("question", "")
                
                # Process options
                options = sample.get("options", {})
                formatted_options = []
                
                # Handle different option formats
                if isinstance(options, dict):
                    # Handle dictionary format options (e.g., {"A": "text", "B": "text"})
                    option_keys = ["A", "B", "C", "D"]
                    for key in option_keys:
                        if key in options:
                            formatted_options.append(options[key])
                elif isinstance(options, list):
                    # Handle list format options
                    formatted_options = options
                
                # Ensure at least 4 options
                while len(formatted_options) < 4:
                    formatted_options.append(f"Choices {chr(65 + len(formatted_options))}")
                
                # Process answer
                answer = sample.get("answer", "")
                answer_idx = sample.get("answer_idx", None)
                
                # Create processed sample
                processed_sample = {
                    "question": question,
                    "options": formatted_options,
                    "answer": answer,
                    "answer_idx": answer_idx,
                    "meta_info": sample.get("meta_info", {}),
                    "metamap_phrases": sample.get("metamap_phrases", [])
                }
                
                processed_data.append(processed_sample)
            except Exception as e:
                logger.error(f"{e}")
        
        return processed_data
    
    def extract_medical_entities(self, text):
        """Extract medical entities by delegating to the KG handler"""
        return self.kg_handler.extract_medical_entities(text)
        
    def retrieve_knowledge_subgraph(self, entities):
        """Retrieve a knowledge subgraph for the given entities"""
        return self.kg_handler.retrieve_knowledge_subgraph(entities)
        
    def format_knowledge_as_context(self, subgraph_info):
        """Format knowledge as text context"""
        if not subgraph_info["entities"] and not subgraph_info["relationships"]:
            return ""
        
        # Use the same implementation as in entity_extraction.py
        context = "Related medical knowledge:\n"
        
        # Organize entities by category
        entities_by_category = {}
        for entity in subgraph_info["entities"]:
            category = entity.get("category", "unknown")
            if category not in entities_by_category:
                entities_by_category[category] = []
            entities_by_category[category].append(entity)
        
        # Add entity information
        for category, entities in entities_by_category.items():
            if entities:
                context += f"\n{category.capitalize()}:\n"
                for entity in entities:
                    entity_name = entity.get("id", "").split(": ")[-1] if ": " in entity.get("id", "") else entity.get("id", "")
                    if not entity_name:
                        continue
                    properties = ", ".join([f"{k}: {v}" for k, v in entity.items() if k not in ["id", "category"] and v])
                    context += f"- {entity_name}" + (f" ({properties})" if properties else "") + "\n"
        
        # Add relationship information
        if subgraph_info["relationships"]:
            context += "\nRelationships:\n"
            for rel in subgraph_info["relationships"]:
                source = rel.get("source", "").split(": ")[-1] if ": " in rel.get("source", "") else rel.get("source", "")
                target = rel.get("target", "").split(": ")[-1] if ": " in rel.get("target", "") else rel.get("target", "")
                rel_type = rel.get("type", "")
                if source and target and rel_type:
                    context += f"- {source} --[{rel_type}]--> {target}\n"
        
        return context
    
    def _get_kg_enhanced_prediction(self, question, options, sample):
        """Get answer prediction using knowledge graph enhancement"""
        
        # Convert options list to dictionary format
        options_dict = {
            "A": options[0],
            "B": options[1],
            "C": options[2],
            "D": options[3]
        }
        
        # Use the KG enhancement approach
        from kg_enhancement import ask_kg_enhanced_question
        
        answer_letter, enhanced_prompt = ask_kg_enhanced_question(
            question=question,
            options=options_dict,
            kg_handler=self.kg_handler,
            llm_interface=self.llm_interface
        )
        
        logger.info(f"Enhanced result: {answer_letter}")
        return answer_letter
    
    def run_evaluation(self, test_data, output_file="kg_enhanced_medqa_results.json", sample_limit=None, entity_log_file="medical_entities_log.json"):
        """Evaluate medical QA data using KG enhancement"""
        # Preprocess data
        processed_data = self.preprocess_data(test_data)
    
        # Optionally limit the number of test samples
        if sample_limit and sample_limit < len(processed_data):
           processed_data = processed_data[:sample_limit]
           logger.info(f"Limited evaluation to {sample_limit} samples")
    
        # Initialize results
        results = []
        correct_count = 0
    
        # Initialize entity records
        entity_records = []
        total_entity_count = 0
        entity_frequency = {}  # For tracking entity occurrence frequency
        entity_by_category = {}  # For categorizing entities
    
        # Process each sample
        for i, sample in enumerate(tqdm(processed_data, desc="Processing")):
            try:
                logger.info(f"Processing question {i+1}/{len(processed_data)}")
            
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
            
                # Record entity data
                entity_record = {
                    "question_id": i,
                    "question_preview": question[:100] + "..." if len(question) > 100 else question,
                    "entities": entities_info,
                    "entity_count": len(entities_info)
                }
                entity_records.append(entity_record)
            
                # Update total entity count
                total_entity_count += len(entities_info)
            
                # Update entity frequency statistics
                for entity in entities_info:
                    entity_name = entity["name"]
                    entity_category = entity["category"]
                
                    # Record entity occurrence frequency
                    if entity_name not in entity_frequency:
                        entity_frequency[entity_name] = {
                            "count": 0, 
                            "category": entity_category
                        }
                    entity_frequency[entity_name]["count"] += 1
                
                    # Categorize by type
                    if entity_category not in entity_by_category:
                        entity_by_category[entity_category] = 0
                    entity_by_category[entity_category] += 1
            
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
                    logger.info(f"Checkpoint {i+1}/{len(processed_data)}. Current accuracy: {accuracy:.2%}")
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
    
        # Save entity records
        entity_summary = {
            "total_questions": len(processed_data),
            "total_entities_extracted": total_entity_count,
            "average_entities_per_question": total_entity_count / len(processed_data) if processed_data else 0,
            "entity_records": entity_records,
            "entity_by_category": {k: v for k, v in sorted(entity_by_category.items(), key=lambda item: item[1], reverse=True)},
            "entity_frequency": [{"name": k, "category": v["category"], "count": v["count"]} 
                            for k, v in sorted(entity_frequency.items(), key=lambda x: x[1]["count"], reverse=True)]
        }
    
        with open(entity_log_file, 'w', encoding='utf-8') as f:
            json.dump(entity_summary, f, indent=2, ensure_ascii=False)
    
        logger.info(f"Entity extraction records saved to: {entity_log_file}")
        logger.info(f"Total of {total_entity_count} medical entities extracted from {len(processed_data)} questions")
    
        # Generate detailed report
        self.generate_detailed_report(results)
    
        logger.info(f"Final accuracy: {final_accuracy:.2%}")
        return final_accuracy, results
    
    def run_baseline_evaluation(self, test_data, output_file="baseline_medqa_results.json", sample_limit=None):
        """Evaluate medical QA data using only GPT-4 baseline method"""
        # Preprocess data to ensure consistent format
        processed_data = self.preprocess_data(test_data)
        
        results = []
        correct_count = 0
        
        # Optionally limit the number of test samples
        if sample_limit and sample_limit < len(processed_data):
            processed_data = processed_data[:sample_limit]
        
        for i, sample in enumerate(tqdm(processed_data, desc="Baseline")):
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
            
            try:
                # Format multiple-choice question with options
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
                
                logger.info(f"Processing baseline question {i+1}/{len(processed_data)}")
                
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
                    predicted_answer = "A"  # Default
                
                # Check if prediction is correct
                is_correct = (predicted_answer == correct_answer)
                if is_correct:
                    correct_count += 1
                
                # Print results
                result_str = f"Baseline: {predicted_answer}" + (" ✓ Correct" if is_correct else f" ✗ Wrong (Correct answer: {correct_answer})")
                logger.info(result_str)
                
                # Calculate current accuracy
                current_accuracy = correct_count / (i + 1)
                logger.info(f"Baseline: {correct_count}/{i+1} = {current_accuracy:.2%}")
                
                # Save result
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
                logger.error(f"{e}")
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
                logger.info(f"Checkpoint {i+1}/{len(processed_data)} Current accuracy: {accuracy_so_far:.2%}")
                
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
    
    def _calculate_correlation(self, x, y):
        """Calculate Pearson correlation coefficient between two lists"""
        if not x or not y or len(x) != len(y) or len(x) < 2:
            return 0
            
        try:
            import numpy as np
            return float(np.corrcoef(x, y)[0, 1])
        except Exception as e:
            logger.error(f"{e}")
            
            # Manual calculation of correlation coefficient
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(i*j for i, j in zip(x, y))
            sum_x2 = sum(i*i for i in x)
            sum_y2 = sum(j*j for j in y)
            
            numerator = n * sum_xy - sum_x * sum_y
            denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5
            
            if denominator == 0:
                return 0
                
            return numerator / denominator

    def analyze_results(self, kg_results_file, baseline_results_file=None):
        """Analyze evaluation results to identify patterns and compare performance"""
        try:
            with open(kg_results_file, 'r', encoding='utf-8') as f:
                kg_data = json.load(f)
                
            if baseline_results_file:
                with open(baseline_results_file, 'r', encoding='utf-8') as f:
                    baseline_data = json.load(f)
            else:
                baseline_data = None
                
            # Get results
            kg_results = kg_data.get('results', [])
            baseline_results = baseline_data.get('results', []) if baseline_data else []
            
            if not kg_results:
                return
                
            # Overall statistics
            kg_total = len(kg_results)
            kg_correct = sum(1 for r in kg_results if r.get('is_correct', False))
            kg_accuracy = kg_correct / kg_total if kg_total > 0 else 0
            
            if baseline_results:
                baseline_total = len(baseline_results)
                baseline_correct = sum(1 for r in baseline_results if r.get('is_correct', False))
                baseline_accuracy = baseline_correct / baseline_total if baseline_total > 0 else 0
                
                # Match paired results for direct comparison
                paired_results = []
                for kg_res in kg_results:
                    for baseline_res in baseline_results:
                        if kg_res.get('id') == baseline_res.get('id'):
                            paired_results.append({
                                'id': kg_res.get('id'),
                                'question': kg_res.get('question'),
                                'kg_correct': kg_res.get('is_correct', False),
                                'baseline_correct': baseline_res.get('is_correct', False)
                            })
                            break
                            
                # Calculate number of questions where KG system is correct but baseline is wrong (improvements)
                kg_improvements = sum(1 for p in paired_results if p['kg_correct'] and not p['baseline_correct'])
                
                # Calculate number of questions where baseline is correct but KG system is wrong (regressions)
                kg_regressions = sum(1 for p in paired_results if not p['kg_correct'] and p['baseline_correct'])

            print("\n=== Result Analysis ===")
            print(f"KG-Enhanced System:")
            print(f"- Total Questions: {kg_total}")
            print(f"- Correct Answers: {kg_correct}")
            print(f"- Accuracy: {kg_accuracy:.2%}")

            if baseline_results:
                print(f"\nBaseline System (GPT-4 Only):")
                print(f"- Total Questions: {baseline_total}")
                print(f"- Correct Answers: {baseline_correct}")
                print(f"- Accuracy: {baseline_accuracy:.2%}")

                print(f"\nDirect Comparison:")
                print(f"- Paired Questions: {len(paired_results)}")
                print(f"- KG Enhancement Improvements: {kg_improvements} questions (KG correct, baseline incorrect)")
                print(f"- KG Enhancement Regressions: {kg_regressions} questions (KG incorrect, baseline correct)")
                print(f"- Net Improvement: {kg_improvements - kg_regressions} questions")

                if kg_accuracy > baseline_accuracy:
                    print(
                        f"- Overall: KG-Enhanced System outperforms the Baseline System by {(kg_accuracy - baseline_accuracy):.2%}")
                elif kg_accuracy < baseline_accuracy:
                    print(
                        f"- Overall: Baseline System outperforms the KG-Enhanced System by {(baseline_accuracy - kg_accuracy):.2%}")
                else:
                    print("- Overall: Both systems perform equally well")

            # Analyze most common wrong answers in KG-enhanced system
            kg_wrong_answers = [r for r in kg_results if not r.get('is_correct', False)]
            if kg_wrong_answers:
                for i, wrong in enumerate(kg_wrong_answers[:5]):  # Show 5 examples
                    q = wrong.get('question', '')
                    q_preview = q[:80] + "..." if len(q) > 80 else q
                    correct = wrong.get('correct_answer', '')
                    predicted = wrong.get('predicted_answer', '')
                    print(f"{i+1}. Q: {q_preview}")
                    print(f"   {correct} vs {predicted}")
                    
            return {
                'kg_accuracy': kg_accuracy,
                'baseline_accuracy': baseline_accuracy if baseline_results else None,
                'improvement': (kg_accuracy - baseline_accuracy) if baseline_results else None
            }
        except Exception as e:
            logger.error(f"Error analyzing results: {e}")
            return None

    def generate_detailed_report(self, results, output_file="detailed_kg_report.json"):
        """Generate detailed evaluation report analyzing knowledge graph contribution"""
        detailed_results = []
        
        for result in results:
            # Get detailed information for the result
            detailed_result = {
                "id": result.get("id"),
                "question": result.get("question"),
                "options": result.get("options"),
                "correct_answer": result.get("correct_answer"),
                "predicted_answer": result.get("predicted_answer"),
                "is_correct": result.get("is_correct"),
                "entities_extracted": result.get("entities_extracted", []),
                "knowledge_used": result.get("knowledge_used", False),
                "knowledge_size": result.get("knowledge_size", 0)
            }
            
            detailed_results.append(detailed_result)
        
        # Group analysis by whether knowledge was used
        with_knowledge = [r for r in detailed_results if r["knowledge_used"]]
        without_knowledge = [r for r in detailed_results if not r["knowledge_used"]]
        
        # Calculate accuracy for each group
        with_knowledge_accuracy = sum(1 for r in with_knowledge if r["is_correct"]) / len(with_knowledge) if with_knowledge else 0
        without_knowledge_accuracy = sum(1 for r in without_knowledge if r["is_correct"]) / len(without_knowledge) if without_knowledge else 0
        
        # Generate report
        report = {
            "detailed_results": detailed_results,
            "summary": {
                "total_questions": len(detailed_results),
                "questions_with_knowledge": len(with_knowledge),
                "questions_without_knowledge": len(without_knowledge),
                "overall_accuracy": sum(1 for r in detailed_results if r["is_correct"]) / len(detailed_results) if detailed_results else 0,
                "with_knowledge_accuracy": with_knowledge_accuracy,
                "without_knowledge_accuracy": without_knowledge_accuracy,
                "knowledge_impact": with_knowledge_accuracy - without_knowledge_accuracy,
                "knowledge_impact_percentage": f"{(with_knowledge_accuracy - without_knowledge_accuracy) * 100:.2f}%",
                "entity_stats": {
                    "average_entities_per_question": sum(len(r["entities_extracted"]) for r in detailed_results) / len(detailed_results) if detailed_results else 0,
                    "max_entities": max((len(r["entities_extracted"]) for r in detailed_results), default=0),
                    "min_entities": min((len(r["entities_extracted"]) for r in detailed_results), default=0)
                },
                "knowledge_size_analysis": {
                    "average_knowledge_size": sum(r["knowledge_size"] for r in with_knowledge) / len(with_knowledge) if with_knowledge else 0,
                    "max_knowledge_size": max((r["knowledge_size"] for r in with_knowledge), default=0),
                    "min_knowledge_size": min((r["knowledge_size"] for r in with_knowledge), default=0),
                    "correlation_with_accuracy": self._calculate_correlation([r["knowledge_size"] for r in with_knowledge], [1 if r["is_correct"] else 0 for r in with_knowledge]) if with_knowledge else 0
                }
            }
        }
        
        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved to: {output_file}")

        # Print summary
        print("\n=== Knowledge Graph Contribution Analysis ===")
        print(f"Total Questions: {report['summary']['total_questions']}")
        print(
            f"Questions Using Knowledge Graph: {report['summary']['questions_with_knowledge']} ({report['summary']['questions_with_knowledge'] / report['summary']['total_questions']:.1%})")
        print(
            f"Questions Without Knowledge Graph: {report['summary']['questions_without_knowledge']} ({report['summary']['questions_without_knowledge'] / report['summary']['total_questions']:.1%})")
        print(f"Overall Accuracy: {report['summary']['overall_accuracy']:.2%}")
        print(f"Accuracy When Using Knowledge Graph: {report['summary']['with_knowledge_accuracy']:.2%}")
        print(f"Accuracy Without Knowledge Graph: {report['summary']['without_knowledge_accuracy']:.2%}")
        print(
            f"Knowledge Graph Contribution: {report['summary']['knowledge_impact']:.2%} (A positive value indicates a beneficial impact)")

        return report