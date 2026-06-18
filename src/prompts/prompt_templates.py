# This file contains the prompt template we send to the LLM.
# The template tells the LLM: "here is some context from documents,
# now answer the user's question based ONLY on that context."


def build_rag_prompt(context_chunks: list, user_question: str) -> str:
    """
    Builds the final prompt that we send to the LLM.
    
    Args:
        context_chunks: list of dicts with 'text' and 'source'
        user_question: the user's question
    
    Returns:
        A formatted prompt string
    """
    # Combine all retrieved chunks into one context block
    context_text = ""
    for i, chunk in enumerate(context_chunks):
        context_text += f"\n--- Chunk {i+1} (from: {chunk['source']}) ---\n"
        context_text += chunk["text"]
        context_text += "\n"
    
    prompt = f"""You are a helpful assistant. Answer the user's question based ONLY on the context provided below.

If the answer is not in the context, say "I could not find this information in the uploaded documents."

Do not make up information.

CONTEXT:
{context_text}

USER QUESTION:
{user_question}

ANSWER:"""
    
    return prompt
