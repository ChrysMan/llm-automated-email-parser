
import os, sys, json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval, ContextualRecallMetric, ContextualRelevancyMetric, FaithfulnessMetric,  AnswerRelevancyMetric
from deepeval.models.llms.gemini_model import GeminiModel
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.logging import LOGGER

load_dotenv()

def create_graph_metrics() -> List[GEval]:
        eval_model = GeminiModel(model=os.getenv("EVAL_LLM_MODEL", "gemini-2.5-flash"), api_key=os.getenv("EVAL_LLM_BINDING_API_KEY"))
        
        # # Checks if the answer contradicts text context OR the entity/relationship triples.
        # faithfulness = GEval(
        #     name="Graph-Faithfulness",
        #     model=eval_model,
        #     criteria="""Determine if the 'actual output' is factually consistent with the information in the 'retrieval context'. 
        #     The 'retrieval context' contains context chunks, entities, and relationships. 
        #     Contradicting any specific claim entity property or relationship triple in the 'retrieval context' must result in a low score.
        #     The score ranges from 0 to 1, where 1 means the 'actual output' is fully supported by the 'retrieval context' without any contradictions, and 0 means it is completely contradicted.
        #     Provide a granular score with two decimal places (e.g., 0.85) to reflect the exact degree of consistency.""",
        #     evaluation_params=[
        #         LLMTestCaseParams.ACTUAL_OUTPUT, 
        #         LLMTestCaseParams.RETRIEVAL_CONTEXT,
        #         #LLMTestCaseParams.INPUT # Used to see if entities/rels are being applied
        #     ]
        # )

        # # Checks if all necessary information from the Ground Truth is found in the RAG Graph data.
        # context_recall = GEval(
        #     name="Graph-Context-Recall",
        #     model=eval_model,
        #     criteria="""Assess if the 'retrieval context' contains all the necessary facts to produce the 'expected output'. 
        #     The 'retrieval context' includes context chunks, entities and relationships; check if the specific triples and claims required for the 'expected output' were successfully retrieved.
        #     The score ranges from 0 to 1, where 1 means the 'retrieval context' contains all the necessary information to fully support the 'expected output', and 0 means it contains none of the necessary information.
        #     Provide a granular score with two decimal places (e.g., 0.85) to reflect the exact degree of consistency.
        #     """,
        #     evaluation_params=[
        #         LLMTestCaseParams.RETRIEVAL_CONTEXT, 
        #         LLMTestCaseParams.EXPECTED_OUTPUT
        #     ]
        # )

        # context_relevancy = GEval(
        #     name="Graph-Context-Relevancy",
        #     model=eval_model,
        #     threshold=0.2,
        #     criteria="""Evaluate the efficiency of the 'retrieval context'. 
        #     Calculate the ratio of claims in the context chunks, retrieved entities and relationships that are actually 
        #     necessary to answer the 'input' compared to the total number of items retrieved. 
        #     Do NOT penalize based on the order of items; only focus on the presence of signal vs. noise.
        #     The score ranges from 0 to 1, where 1 means all retrieved items are relevant and necessary to answer the question, and 0 means none of the retrieved items are relevant.
        #     Provide a granular score with two decimal places (e.g., 0.85) to reflect the exact degree of consistency.
        #     """,
        #     evaluation_params=[
        #         LLMTestCaseParams.INPUT, 
        #         LLMTestCaseParams.RETRIEVAL_CONTEXT
        #     ]
        # )

        faithfulness = FaithfulnessMetric(
            threshold=0.6,
            model=eval_model,
            include_reason=True
        )

        answer_relevancy = AnswerRelevancyMetric(
            threshold=0.6,
            model=eval_model,
            include_reason=True
        )
        
        context_recall = ContextualRecallMetric(
            threshold=0.6,
            model=eval_model,
            include_reason=True
        )

        return [faithfulness, answer_relevancy, context_recall]

class GRAPH_RAGEvaluator:
    """
    Evaluator for RAG system outputs using G-Eval framework.
    Evaluates the quality of retrieved contexts and their impact on answer generation.
    """
    def __init__(self, metrics: List[GEval], test_dataset_path: str = None, rag_api_url: str = None):
        self.metrics = metrics
        self.test_dataset_path = test_dataset_path
        self.rag_api_url = rag_api_url
        
        if test_dataset_path is None:
            test_dataset_path = Path(__file__).parent / "sample_dataset.json"

        if rag_api_url is None:
            rag_api_url = os.getenv("RAG_API_URL", "http://localhost:9621")

        self.test_dataset_path = Path(test_dataset_path)
        self.rag_api_url = rag_api_url.rstrip("/")
        self.results_dir = Path(__file__).parent / "lightrag_rag_results"
        self.results_dir.mkdir(exist_ok=True)

        # Load test dataset
        self.test_cases = self._load_test_dataset()


    def _load_test_dataset(self) -> List[Dict[str, str]]:
        """Load test cases from JSON file"""
        if not self.test_dataset_path.exists():
            raise FileNotFoundError(f"Test dataset not found: {self.test_dataset_path}")

        with open(self.test_dataset_path) as f:
            data = json.load(f)

        return data.get("test_cases", [])
    
    def generate_rag_response(self, question: str) -> Dict[str, Any]:
        """Send query to RAG API and get response with retrieved contexts"""
        import requests

        try:
            payload = {"query": question}
            response = requests.post(
                f"{self.rag_api_url}/query", 
                json=payload
            )
            response.raise_for_status()
            result = response.json()

            answer = result.get("answer", "No response generated")
            context = result.get("retrieved_contexts", {})

            return {
                "content": answer.get("content", ""),
                "chunks": context.get("chunks", []),
                "entities": context.get("entities", []),
                "relationships": context.get("relationships", [])
            }
        except Exception as e:
            raise Exception(f"Error calling LightRAG API: {type(e).__name__}: {str(e)}")

    
    def evaluate_single_case(self, idx: int, test_case: Dict[str, str]) -> Dict[str, Any]:
        """Evaluate a single test case and return results"""
        try:
            rag_response = self.generate_rag_response(test_case["question"])
        except Exception as e:
                LOGGER.error("Error generating response for test %s: %s", idx, str(e))
                return {
                    "idx": idx,
                    "question": test_case.get("question", ""),
                    "answer": "",
                    "expected_output": test_case.get("ground_truth", ""),
                    "metrics": {}
                }
        
        if "error" in rag_response:
            return {"error": rag_response["error"]}

        combined_context = [
            f"Context chunks: {rag_response.get('chunks', [])}",
            f"Entities: {rag_response.get('entities', [])}",
            f"Relationships: {rag_response.get('relationships', [])}"
]
        # Combine RAG response with test case for evaluation
        combined_test_case = LLMTestCase(
            input=test_case.get("question", ""),
            actual_output=str(rag_response.get("content", "")),
            retrieval_context=combined_context,
            expected_output=test_case.get("ground_truth", "")
        )

        evaluation_results = evaluate([combined_test_case], self.metrics)
        
        metrics_summary = {}
        for result in evaluation_results.test_results:
            for metric in result.metrics_data:
                clean_name = metric.name.split(" [")[0]

                summary = {
                    "score": metric.score,
                    "success": "✅" if metric.success else "❌",
                    "reason": metric.reason,
                    "model": metric.evaluation_model
                }
                metrics_summary[clean_name] = summary
            
       #print(f"Evaluation results for test case {idx}: {evaluation_results}\n\nType: {type(evaluation_results)}")
        result={
            "idx": idx,
            "question": test_case.get("question", ""),
            "answer": combined_test_case.actual_output,
            "expected_output": test_case.get("ground_truth", ""),
            "metrics": metrics_summary
            }
        return result


if __name__ == "__main__":
    metrics = create_graph_metrics()
    evaluator = GRAPH_RAGEvaluator(metrics=metrics, test_dataset_path="sample_dataset.json", rag_api_url=os.getenv("RAG_API_URL"))

    results = []
    totals = {"Faithfulness": 0.0, "Answer Relevancy": 0.0, "Contextual Recall": 0.0}

    for idx, test_case in enumerate(evaluator.test_cases):
        LOGGER.info(f"Evaluating test case {idx}")
        
        result = evaluator.evaluate_single_case(idx, test_case)
        results.append(result)
        
        metrics_output = result.get("metrics", {})
        
        for m_name in totals.keys():
            score = metrics_output.get(m_name, {}).get("score", 0)
            totals[m_name] += score

    num_cases = len(evaluator.test_cases) if len(evaluator.test_cases) > 0 else 1
    average_scores = {
        name: round(total / num_cases, 3) 
        for name, total in totals.items()
    }

    average_scores["Overall_Score"] = round(sum(average_scores.values()) / len(totals), 3)

    final_output = {
        "individual_results": results,
        "summary_statistics": average_scores
    }

    results_path = evaluator.results_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(final_output, file, indent=2, ensure_ascii=False, default=str)
    LOGGER.info("Evaluation completed. Results saved to %s", results_path)
    