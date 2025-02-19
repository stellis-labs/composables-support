from src.chunking import chunk_text, rank_chunks
from src.summarization import summarize_chunks, recursive_summarization
from src.expansion import expand_summary

def hierarchical_processing_pipeline(problem_statement, solutions):
    """Executes hierarchical processing pipeline using Groq API."""
    
    # Step 1: Chunking
    chunks = [chunk_text(solution) for solution in solutions]
    flattened_chunks = [chunk for sublist in chunks for chunk in sublist]  # Flatten list

    # Step 2: Relevance Filtering
    ranked_chunks = rank_chunks(problem_statement, flattened_chunks)

    # Step 3: Multi-Stage Summarization
    summarized_chunks = summarize_chunks(ranked_chunks)
    final_summary = recursive_summarization(summarized_chunks)

    # Step 4: Structured Expansion
    expanded_report = expand_summary(final_summary, ranked_chunks)

    return expanded_report
