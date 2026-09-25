import os
import pandas as pd
from dotenv import load_dotenv

from phoenix.client import Client
from phoenix.client.types.spans import SpanQuery
from phoenix.evals.llm import LLM
from phoenix.evals.metrics import RetrievalRelevanceEvaluator
from phoenix.evals import evaluate_dataframe, create_classifier

load_dotenv()

def run_evaluations():
    print("--- STARTING LLMOps CONTINUOUS EVALUATION ---")
    
    client = Client()
    eval_llm = LLM(
        provider="openai",
        model="openai/gpt-4o-mini",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )
    
    # ---------------------------------------------------------
    # 1. RETRIEVER EVALUATION (DOCUMENT LEVEL)
    # ---------------------------------------------------------
    retriever_query = SpanQuery().where("span_kind == 'RETRIEVER'").select(
        "input.value"
    ).explode(
        "retrieval.documents", 
        context="document.content"
    )
    
    retriever_df = client.spans.get_spans_dataframe(
        query=retriever_query, 
        project_name="acme-crag-pipeline"
    )
    
    if not retriever_df.empty:
        retriever_df.rename(columns={"input.value": "input"}, inplace=True)
        retriever_df.dropna(subset=['input', 'context'], inplace=True)
        
        # CRITICAL: Rename the index level 'context.span_id' to 'span_id' so Phoenix recognizes it
        retriever_df.index.names = [
            'span_id' if name == 'context.span_id' else name 
            for name in retriever_df.index.names
        ]
        
        print(f"Evaluating {len(retriever_df)} individual documents for Relevance...")
        
        relevance_eval = RetrievalRelevanceEvaluator(llm=eval_llm)
        relevance_results = evaluate_dataframe(
            dataframe=retriever_df,
            evaluators=[relevance_eval]
        )
        
        # NATIVE PANDAS BYPASS: Unpack the actual dict column into label/score/explanation
        relevance_annotations = relevance_results['retrieval_relevance_score'].apply(pd.Series)
        
        client.spans.log_document_annotations_dataframe(
            dataframe=relevance_annotations,
            annotation_name="relevance",
            annotator_kind="LLM"
        )
        print("✓ Document-level Relevance annotations logged.")

    # ---------------------------------------------------------
    # 2. GENERATOR EVALUATION (SPAN LEVEL)
    # ---------------------------------------------------------
    generator_query = SpanQuery().where("span_kind == 'LLM'").select(
        "input.value",
        "output.value"
    )
    
    generator_df = client.spans.get_spans_dataframe(
        query=generator_query, 
        project_name="acme-crag-pipeline"
    )
    
    if not generator_df.empty:
        generator_df.rename(columns={
            "input.value": "input",       
            "output.value": "output"      
        }, inplace=True)
        generator_df.dropna(subset=['input', 'output'], inplace=True)
        
        # CRITICAL: Rename the index level here too
        generator_df.index.names = [
            'span_id' if name == 'context.span_id' else name 
            for name in generator_df.index.names
        ]
        
        print(f"Evaluating {len(generator_df)} LLM spans for Faithfulness...")
        
        # We use a custom classifier because our 'input' contains both the prompt and the context
        faithfulness_prompt = """\
Determine whether the AI's Answer is factually supported by the Context provided in its Prompt.
Prompt (contains context): {input}
AI Answer: {output}
Rule: Reply with exactly one word: 'faithful' or 'unfaithful'."""

        faithfulness_eval = create_classifier(
            name="faithfulness",
            prompt_template=faithfulness_prompt,
            llm=eval_llm,
            choices={"faithful": 1.0, "unfaithful": 0.0},
        )
        
        faithfulness_results = evaluate_dataframe(
            dataframe=generator_df,
            evaluators=[faithfulness_eval]
        )
        
        faithfulness_annotations = faithfulness_results['faithfulness_score'].apply(pd.Series)
        
        client.spans.log_span_annotations_dataframe(
            dataframe=faithfulness_annotations,
            annotation_name="faithfulness",
            annotator_kind="LLM"
        )
        print("✓ Span-level Faithfulness annotations logged.")

if __name__ == "__main__":
    run_evaluations()