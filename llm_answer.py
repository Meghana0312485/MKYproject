from transformers import pipeline

# Load IBM Granite text-generation pipeline
granite_pipe = pipeline(
    "text-generation",
    model="ibm-granite/granite-3.3-2b-instruct",
    tokenizer="ibm-granite/granite-3.3-2b-instruct"
)

def generate_answer(question, retrieved_chunks):
    """
    Generates an answer using IBM Granite model.
    The answer is grounded only on retrieved PDF context.
    """
    
    # Handle empty context safely
    if not retrieved_chunks:
        return "No relevant context found in the PDFs."

    # Combine retrieved chunks into a single context
    context = "\n\n".join(retrieved_chunks)

    # Chat-style prompt converted into a plain string prompt
    prompt = (
        "You are StudyMate, an academic assistant.\n"
        "Provide a clear, correct answer based ONLY on the context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    # Model inference
    output = granite_pipe(
        prompt,
        max_new_tokens=250,
        do_sample=False,
        truncation=True
    )

    # The model outputs a list of dicts → take the generated text
    generated = output[0]["generated_text"]

    # Remove the original prompt from the output
    answer = generated.replace(prompt, "").strip()

    return answer
