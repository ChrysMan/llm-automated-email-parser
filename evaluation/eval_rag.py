import asyncio
from lightrag.evaluation.eval_rag_quality import RAGEvaluator
from dotenv import load_dotenv

load_dotenv()

async def main():
    evaluator = RAGEvaluator(
        test_dataset_path="sample_dataset.json",
        rag_api_url="http://localhost:8080/"
    )
    await evaluator.run()

if __name__ == "__main__":
    asyncio.run(main())