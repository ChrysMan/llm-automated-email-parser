import os, json
import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict
from pathlib import Path
from datetime import datetime

from ragas.metrics.collections import (
    ExactMatch,
    FactualCorrectness,
    SemanticSimilarity,
    CHRFScore,
    NonLLMStringSimilarity,
    DistanceMeasure
)
from google import genai
from ragas.llms import llm_factory
from ragas.embeddings import GoogleEmbeddings
#from ..src.utils.file_io import read_json_file

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)

# -------------------------
# Main evaluator
# -------------------------
class PreprocessingEvaluator:
    """
    Full-file evaluation of preprocessing using RAGAS metrics only.
    """

    def __init__(self, alignment_threshold: float = 0.7):
        eval_model = os.getenv("EVAL_LLM_MODEL", "gpt-4o-mini")
        eval_embedding_model = os.getenv("EVAL_EMBEDDING_MODEL")
        
        client = genai.Client(api_key=os.environ.get("EVAL_LLM_BINDING_API_KEY"))
        
        self.eval_embeddings = GoogleEmbeddings(client=client, model=eval_embedding_model)
        
        self.eval_llm = llm_factory(eval_model, provider="google", client=client)

        self.alignment_threshold = alignment_threshold

        # RAGAS metrics
        self.exact_match = ExactMatch()
        self.factual = FactualCorrectness(llm=self.eval_llm)
        self.semantic = SemanticSimilarity(embeddings=self.eval_embeddings)
        self.chrf = CHRFScore()
        # self.string_sim = NonLLMStringSimilarity(distance_measure=DistanceMeasure.LEVENSHTEIN)


    def _align_emails(self, predicted: List[dict], ground_truth: List[dict]) -> Dict[int, Optional[int]]:
        """
        Align predicted emails to ground truth emails.
        Multiple predicted emails may align to the same GT email.
        """
        alignment = {}

        p_txt = [f"from: {item.get('from','')}\nsent: {item.get('sent','')}\nto:{item.get('to','')}\ncc:{item.get('cc','')}\nsubject:{item.get('subject','')}\nbody:{item.get('body','')}" 
                    for item in predicted]
        gt_txt = [f"from: {item.get('from','')}\nsent: {item.get('sent','')}\nto:{item.get('to','')}\ncc:{item.get('cc','')}\nsubject:{item.get('subject','')}\nbody:{item.get('body','')}" 
                    for item in ground_truth]
        
        for p_idx, p_text in enumerate(p_txt):
            best_score = 0.0
            best_gt = None

            for g_idx, g_text in enumerate(gt_txt):
                score = self.chrf.score(
                    response=p_text,
                    reference=g_text,
                )

                if score > best_score:
                    best_score = score
                    best_gt = g_idx

            alignment[p_idx] = best_gt if best_score >= self.alignment_threshold else None

        return alignment

    def evaluate_headers(self, pr: Dict, gt: Dict) -> float:
        fields = ["from", "to", "cc", "subject"]
        scores = []

        scores = [self.exact_match(response=pr[f], reference=gt[f]) for f in fields]
        
        return float(np.mean(scores)) if scores else 0.0
    
    # -------------------------
    # Content & formatting
    # -------------------------
    def evaluate_content(
        self,
        predicted: List[dict],
        ground_truth: List[dict],
        alignment: Dict[int, Optional[int]],
    ) -> Dict[str, float]:

        scores = defaultdict(list)

        for p_idx, g_idx in alignment.items():
            if g_idx is None:
                continue

            p = predicted[p_idx]
            g = ground_truth[g_idx]

            scores["exact_match"].append(
                self.evaluate_headers(p, g)
            )
            scores["factual_correctness"].append(
                self.factual.score(response=p["body"], reference=g["body"])
            )
            scores["string_similarity"].append(
                self.semantic.score(response=p["body"], reference=g["body"])
            )

        return {
            k: float(np.mean(v)) if v else 0.0
            for k, v in scores.items()
        }

    # -------------------------
    # Deduplication evaluation
    # -------------------------
    def evaluate_deduplication(
        self,
        predicted: List[dict],
        ground_truth: List[dict],
        alignment: Dict[int, Optional[int]],
    ) -> Dict[str, float]:

        gt_to_pred = defaultdict(list)
        for p_idx, g_idx in alignment.items():
            if g_idx is not None:
                gt_to_pred[g_idx].append(p_idx)

        matched_gt = len(gt_to_pred)
        total_gt = len(ground_truth)
        total_pred = len(predicted)

        precision = matched_gt / total_pred if total_pred else 0.0
        recall = matched_gt / total_gt if total_gt else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        return {
            "dedup_precision": precision,                   # Did you remove duplicates correctly?
            "dedup_recall": recall,                         # Did you accidentally delete real emails?
            "dedup_f1": f1,                                 # Overall deduplication quality
            # "extra_duplicates": total_pred - matched_gt,
            # "missed_emails": total_gt - matched_gt,
        }

    # -------------------------
    # Full evaluation
    # -------------------------
    def evaluate(
        self,
        predicted_emails: List[dict],
        ground_truth_emails: List[dict],
    ) -> Dict[str, float]:

        results = {}

        alignment = self._align_emails(predicted_emails, ground_truth_emails)

        results.update(
            self.evaluate_content(
                predicted_emails,
                ground_truth_emails,
                alignment,
            )
        )

        results.update(
            self.evaluate_deduplication(
                predicted_emails,
                ground_truth_emails,
                alignment,
            )
        )

        meaning = np.mean([
            results.get("factual_correctness", 0.0),
            results.get("semantic_similarity", 0.0),
        ])

        overall = (
            0.45 * meaning +
            0.35 * results.get("exact_match", 0.0) +
            0.20 * results.get("dedup_f1", 0.0)
        )

        results.update({"overall_score": overall})

        return results

def main():
    results_dir = Path(__file__).parent / "results"
    json_path = (
            results_dir
            / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
    
    evaluator = PreprocessingEvaluator()

    with open("/home/chryssida/DATA_TUC-KRITI/TRUCK EXPORT/244037/244037_unique.json", "r", encoding="utf-8") as f:
            predictions = json.load(f)
    with open("/home/chryssida/DATA_TUC-KRITI/TRUCK EXPORT/244037/244037_gt.json", "r", encoding="utf-8") as f:
            ground_truth = json.load(f)
    #ground_truth = read_json_file("/home/chryssida/DATA_TUC-KRITI/TRUCK EXPORT/244037/244037_gt.json")
    try:
        results = evaluator.evaluate(predictions, ground_truth)

        with open(json_path, "a") as f:
            json.dump(results, f, indent=2)

        print("Success")
    except Exception as e:
        print(f"Error: {e}")
    
if __name__ == "__main__":
    main()