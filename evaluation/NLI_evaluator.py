from sentence_transformers import CrossEncoder
import numpy as np
import torch


class NLIEvaluator:
    def __init__(self, model_name='cross-encoder/nli-deberta-v3-base'):
         self.model = CrossEncoder(model_name)

    def compute_score(self, response: str, reference: str) -> float:

        probabilities = self.model.predict([(reference, response)],apply_softmax=True)[0]

        contradiction_prob = probabilities[0]   # False info: Measures critical changes like dates, numbers, names.
        entailment_prob = probabilities[1]      # True info: Measures if all core facts are kept. Also it understands if the meaning is the same
        neutral_prob = probabilities[2]         # Rephrased info: Measures the neutrality of the rephrase.

        raw_score = entailment_prob + 0.5 * neutral_prob
        #print(f"entailment: {entailment_prob}, neutral: {neutral_prob}, contradiction: {contradiction_prob}, raw_score: {raw_score}")
        #final_score = raw_score / 1.5
        print("\nfinal_score:", raw_score)
        return max(0.0, min(1.0, float(raw_score)))