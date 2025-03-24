import json
from config import logger

def _calculate_correlation(self, x, y):
    if not x or not y or len(x) != len(y) or len(x) < 2:
        return 0
        
    try:
        import numpy as np
        return float(np.corrcoef(x, y)[0, 1])
    except Exception as e:
        logger.error(f"Error while loading: {e}")

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
            logger.warning("No results found.")
            return

        kg_total = len(kg_results)
        kg_correct = sum(1 for r in kg_results if r.get('is_correct', False))
        kg_accuracy = kg_correct / kg_total if kg_total > 0 else 0
        
        if baseline_results:
            baseline_total = len(baseline_results)
            baseline_correct = sum(1 for r in baseline_results if r.get('is_correct', False))
            baseline_accuracy = baseline_correct / baseline_total if baseline_total > 0 else 0

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

            kg_improvements = sum(1 for p in paired_results if p['kg_correct'] and not p['baseline_correct'])

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

        kg_wrong_answers = [r for r in kg_results if not r.get('is_correct', False)]
        if kg_wrong_answers:
            for i, wrong in enumerate(kg_wrong_answers[:5]):
                q = wrong.get('question', '')
                q_preview = q[:80] + "..." if len(q) > 80 else q
                correct = wrong.get('correct_answer', '')
                predicted = wrong.get('predicted_answer', '')
                print(f"{i+1}. Q: {q_preview}")
                print(f"   Correct: {correct}, Predict: {predicted}")
                
        return {
            'kg_accuracy': kg_accuracy,
            'baseline_accuracy': baseline_accuracy if baseline_results else None,
            'improvement': (kg_accuracy - baseline_accuracy) if baseline_results else None
        }
    except Exception as e:
        logger.error(f"Error while analyzing: {e}")
        return None

def generate_detailed_report(self, results, output_file="detailed_kg_report.json"):
    detailed_results = []
    
    for result in results:
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

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Report saved to: {output_file}")

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