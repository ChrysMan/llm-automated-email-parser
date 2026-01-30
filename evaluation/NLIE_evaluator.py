from sentence_transformers import CrossEncoder
import numpy as np

class NLIEvaluator:
    def __init__(self, model_name='cross-encoder/nli-deberta-v3-base'):
         self.model = CrossEncoder(model_name)

    def compute_score(self, response: str, reference: str) -> float:

        scores = self.model.predict([(reference, response)])

        exp_scores = np.exp(scores[0])
        probabilities = exp_scores / np.sum(exp_scores)
        
        contradiction_prob = probabilities[0]   # True info: Measures if all core facts are kept. Also it understands if the meaning is the same
        entailment_prob = probabilities[1]      # Rephrased info: Measures if the neutrality of the rephrase.
        neutral_prob = probabilities[2]         # False info: Measures critical changes like dates, numbers, names..


        final_score = entailment_prob + (0.7 * neutral_prob) - (1.5 * contradiction_prob) # We mostly care if important info is missing not if extra text is added (e.g. trailing text)
        print("final_score:", final_score)
        return max(0.0, min(1.0, float(final_score)))