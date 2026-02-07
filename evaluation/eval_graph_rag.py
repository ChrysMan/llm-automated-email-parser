import os, sys, json
from pathlib import Path
from typing import List, Dict, Optional, Any
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.models.llms.gemini_model import GeminiModel
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.logging import LOGGER

load_dotenv()

def create_graph_metrics() -> List[GEval]:
        eval_model = GeminiModel(model=os.getenv("EVAL_LLM_MODEL", "gemini-2.5-flash"), api_key=os.getenv("EVAL_LLM_BINDING_API_KEY"))
        
        # Checks if the answer contradicts text context OR the entity/relationship triples.
        faithfulness = GEval(
            name="Graph-Faithfulness",
            model=eval_model,
            criteria="""Determine if the 'actual output' is factually consistent with the information in the 'retrieval context'. 
            The 'retrieval context' contains context chunks, entities, and relationships. 
            Contradicting any specific claim entity property or relationship triple in the 'retrieval context' must result in a low score.""",
            evaluation_params=[
                LLMTestCaseParams.ACTUAL_OUTPUT, 
                LLMTestCaseParams.RETRIEVAL_CONTEXT,
                LLMTestCaseParams.INPUT # Used to see if entities/rels are being applied
            ]
        )

        # Checks if all necessary information from the Ground Truth is found in the RAG Graph data.
        context_recall = GEval(
            name="Graph-Context-Recall",
            model=eval_model,
            criteria="""Assess if the 'retrieval context' contains all the necessary facts to produce the 'expected output'. 
            The 'retrieval context' includes context chunks, entities and relationships; check if the specific triples and claims required for the 'expected output' were successfully retrieved.""",
            evaluation_params=[
                LLMTestCaseParams.RETRIEVAL_CONTEXT, 
                LLMTestCaseParams.EXPECTED_OUTPUT
            ]
        )

        context_relevancy = GEval(
            name="Graph-Context-Relevancy",
            model=eval_model,
            criteria="""Evaluate the efficiency of the 'retrieval context'. 
            Calculate the ratio of claims in the context chunks, retrieved entities and relationships that are actually 
            necessary to answer the 'input' compared to the total number of items retrieved. 
            Do NOT penalize based on the order of items; only focus on the presence of signal vs. noise.""",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT]
        )

        return [faithfulness, context_recall, context_relevancy]

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
        self.results_dir = Path(__file__).parent / "graph_rag_results"
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

            return response.json() if response.status_code == 200 else {"error": f"Failed to retrieve response: {response.status_code}"}
        except Exception as e:
            raise Exception(f"Error calling LightRAG API: {type(e).__name__}: {str(e)}")

    
    def evaluate_single_case(self, idx: int, test_case: Dict[str, str]) -> Dict[str, Any]:
        """Evaluate a single test case and return results"""
        try:
            rag_response = self.generate_rag_response(test_case["question"])
        except Exception as e:
                LOGGER.error("Error generating response for test %s: %s", idx, str(e))
                return {
                    "question": test_case["question"],
                    "answer": "",
                    "expected_output": test_case["ground_truth"],
                    "metrics": {},
                    "ragas_score": 0
                }
        
        if "error" in rag_response:
            return {"error": rag_response["error"]}

        combined_context = [
            f"Context chunks: {rag_response.get('data', {}).get('chunks', [])}",
            f"Entities: {rag_response.get('data', {}).get('entities', [])}",
            f"Relationships: {rag_response.get('data', {}).get('relationships', [])}"
]
        # Combine RAG response with test case for evaluation
        combined_test_case = LLMTestCase(
            input=test_case["question"],
            actual_output=rag_response.get("llm_response", {}),
            retrieval_context=combined_context,
            expected_output=test_case["ground_truth"]
        )

        evaluation_results = evaluate([combined_test_case], self.metrics)

        result={
            "idx": idx,
            "question": test_case.input,
            "answer": combined_test_case.actual_output,
            "expected_output": test_case.expected_output,
            "metrics": evaluation_results[0].test_results if evaluation_results else {},
            "overall_score": sum(evaluation_results[0].test_results.values()) / len(evaluation_results[0].test_results) if evaluation_results and evaluation_results[0].test_results else 0
            }
        return result


if __name__ == "__main__":
    metrics = create_graph_metrics()
    evaluator = GRAPH_RAGEvaluator(metrics=metrics, test_dataset_path="sample_dataset.json", rag_api_url=os.getenv("RAG_API_URL"))

    results = []
    for idx, test_case in enumerate(evaluator.test_cases):
        LOGGER.info("Evaluating test case %d: %s", idx, test_case.get("input", ""))
        result = evaluator.evaluate_single_case(idx, test_case)
        results.append(result)

    # Save results to JSON
    results_path = evaluator.results_dir / f"graph_rag_evaluation_{len(results)}.json"
    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False, default=str)
    LOGGER.info("Evaluation completed. Results saved to %s", results_path)
    