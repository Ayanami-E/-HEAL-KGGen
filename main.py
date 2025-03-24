from config import logger, KG_JSON_PATH, client
from data_utils import load_medqa_data, show_sample
from kg_handler import JSONKnowledgeGraphHandler
from llm_interface import LLMInterface
# Import the improved framework class
from medqa_core import MedQATestFramework
import json
import argparse
import os
from datetime import datetime
import traceback

# Main program entry
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Medical QA Knowledge Graph Enhanced Evaluation System')
    parser.add_argument('--test_file', type=str, 
                        default=r".\Task5_cleaned.json",
                        help='Test data file path')
    parser.add_argument('--kg_path', type=str, default=KG_JSON_PATH,
                        help='Knowledge graph file path')
    parser.add_argument('--sample_limit', type=int, default=148,
                        help='Limit of samples to process')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Results output directory')
    parser.add_argument('--mode', type=str, choices=['kg', 'baseline', 'both'], default='both',
                        help='Evaluation mode: kg=knowledge graph enhanced, baseline=base model, both=evaluate both')
    parser.add_argument('--entity_log', type=str, default='medical_entities_log.json',
                       help='Entity extraction log file path')
    
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set output file paths
    kg_output_file = os.path.join(output_dir, "kg_enhanced_results.json")
    baseline_output_file = os.path.join(output_dir, "baseline_results.json")
    report_output_file = os.path.join(output_dir, "comparison_report.json")

    # Load test dataset
    logger.info(f"Loading test data: {args.test_file}")
    test_data = load_medqa_data(args.test_file)
    logger.info(f"Loaded {len(test_data)} test samples")

    # Display sample information
    if test_data:
        show_sample(test_data)

        # Check and log some dataset statistics
        question_lengths = [len(sample.get('question', '')) for sample in test_data]
        option_counts = [len(sample.get('options', {})) for sample in test_data]

        logger.info("Dataset Statistics:")
        logger.info(f"- Average question length: {sum(question_lengths) / len(question_lengths):.1f} characters")
        logger.info(f"- Maximum question length: {max(question_lengths)} characters")
        logger.info(f"- Option count statistics: Minimum {min(option_counts)}, Maximum {max(option_counts)}")
    
    # Initialize evaluation framework
    try:
        # Initialize knowledge graph
        kg_handler = JSONKnowledgeGraphHandler(args.kg_path)
        
        # Initialize LLM interface
        llm_interface = LLMInterface(client=client)
        
        # Initialize evaluation framework
        test_framework = MedQATestFramework(kg_handler=kg_handler, llm_interface=llm_interface)
        
        # Run evaluation
        results = {}

        # KG-enhanced evaluation
        if args.mode in ['kg', 'both']:
            logger.info("\n=== Running Knowledge Graph Enhanced Evaluation ===")
            kg_accuracy, kg_results = test_framework.run_evaluation(
                test_data,
                output_file=kg_output_file,
                sample_limit=args.sample_limit,
                entity_log_file=os.path.join(output_dir, args.entity_log)
            )
            results['kg'] = {
                'accuracy': kg_accuracy,
                'results': kg_results
            }
            logger.info(f"Knowledge Graph Enhanced Evaluation Completed, Accuracy: {kg_accuracy:.2%}")

        # Baseline evaluation
        if args.mode in ['baseline', 'both']:
            logger.info("\n=== Running Baseline Model Evaluation ===")
            baseline_accuracy, baseline_results = test_framework.run_baseline_evaluation(
                test_data,
                output_file=baseline_output_file,
                sample_limit=args.sample_limit
            )
            results['baseline'] = {
                'accuracy': baseline_accuracy,
                'results': baseline_results
            }
            logger.info(f"Baseline Model Evaluation Completed, Accuracy: {baseline_accuracy:.2%}")

        # Analyze and compare results
        if args.mode == 'both':
            logger.info("\n=== Analyzing Comparison Results ===")
            analysis = test_framework.analyze_results(kg_output_file, baseline_output_file)

            # Save analysis report
            with open(report_output_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)

            logger.info(f"Analysis report saved to: {report_output_file}")

            # Print final results
            print("\n=== Final Evaluation Results ===")
            print(f"- Knowledge Graph Enhanced System Accuracy: {kg_accuracy:.2%}")
            print(f"- Baseline System Accuracy: {baseline_accuracy:.2%}")
            print(f"- Improvement Effect: {kg_accuracy - baseline_accuracy:.2%}")

            if kg_accuracy > baseline_accuracy:
                print(f"- Conclusion: The Knowledge Graph Enhanced System outperforms the Baseline System")
            elif kg_accuracy < baseline_accuracy:
                print(f"- Conclusion: The Baseline System outperforms the Knowledge Graph Enhanced System")
            else:
                print(f"- Conclusion: Both systems perform equally well")

        logger.info(f"All evaluations completed, results saved in {output_dir} directory")

    except Exception as e:
        logger.error(f"Error occurred during evaluation: {e}")
        logger.error(traceback.format_exc())

    return 0

if __name__ == "__main__":
    main()