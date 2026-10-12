# V2: Retrieval-Augmented Generation

V2 extends the V1 SciFact dense-retrieval baseline into a complete RAG pipeline.

## Flow

```text
Question
  -> Sentence Transformer query embedding
  -> FAISS top-k retrieval
  -> Retrieved SciFact evidence
  -> Context construction
  -> LLM prompt
  -> Grounded answer with source citations
```

## Design principle

The LLM does **not** search SciFact. Retrieval and generation are separate stages. FAISS retrieves evidence first; only that retrieved evidence is supplied to the LLM.

## Grounding rules

The generation prompt instructs the model to:

- use only retrieved sources;
- cite source numbers;
- state when evidence is insufficient;
- avoid silently adding outside knowledge.

## Logging

Each run is appended to `results/rag_runs.jsonl` with the query, retrieved document IDs/titles/scores, model name, generation time, and generated answer.

## V2 success criteria

1. Retrieval returns relevant SciFact evidence.
2. The LLM produces an answer grounded in that evidence.
3. The answer cites retrieved source numbers.
4. Unsupported questions are handled with an explicit insufficiency statement.
5. Runs are saved for later qualitative and quantitative evaluation.
