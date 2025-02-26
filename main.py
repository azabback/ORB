import os
import fitz  # PyMuPDF for PDF reading
import cohere
from mistralai.client import MistralClient
import google.generativeai as genai  # Gemini SDK
from dotenv import load_dotenv
from verifier import verify_llm_responses

# Load API Keys
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not COHERE_API_KEY or not MISTRAL_API_KEY or not GEMINI_API_KEY:
    raise ValueError("Missing API keys. Please check your .env file.")

# Initialize API Clients
cohere_client = cohere.Client(COHERE_API_KEY)
mistral_client = MistralClient(api_key=MISTRAL_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)  # Gemini Initialization

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Error: PDF file not found at '{pdf_path}'")
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text

# Chunk long text
def chunk_text(text, chunk_size=10000):
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]

# Summarization with Cohere
def summarize_with_cohere(text, model="command-r-plus"):
    prompt = f"Summarize the following research paper:\n\n{text[:3000]}..."
    response = cohere_client.generate(
        model=model, prompt=prompt, max_tokens=300
    )
    return response.generations[0].text.strip()

# Summarization with Mistral
def summarize_with_mistral(text, model="mistral-small"):
    shortened_text = text[:15000]
    prompt = f"Summarize the following research paper:\n\n{shortened_text}..."
    messages = [{"role": "user", "content": prompt}]
    try:
        response = mistral_client.chat(model=model, messages=messages)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error summarizing with Mistral: {str(e)}"

# Summarization with Gemini
def summarize_with_gemini(text):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(f"Summarize the following research paper:\n\n{text[:3000]}...")
    return response.text.strip() if response and response.text else "Error: No response from Gemini."

# Multi-step summarization for long texts
def summarize_long_text(text, model_choice):
    chunks = list(chunk_text(text, chunk_size=10000))
    partial_summaries = []
    for chunk in chunks:
        if model_choice == "Mistral":
            summary = summarize_with_mistral(chunk)
        elif model_choice == "Gemini":
            summary = summarize_with_gemini(chunk)
        else:
            summary = summarize_with_cohere(chunk)
        partial_summaries.append(summary)
    combined_text = "\n".join(partial_summaries)
    final_summary = summarize_with_mistral(combined_text) if model_choice == "Mistral" else summarize_with_gemini(combined_text)
    return final_summary

# Answering questions with Cohere
def answer_question_with_cohere(text, question, model="command-r-plus"):
    prompt = f"The following is a research paper:\n\n{text[:4000]}...\n\nAnswer this question: {question}"
    response = cohere_client.generate(
        model=model, prompt=prompt, max_tokens=300
    )
    return response.generations[0].text.strip()

# Answering questions with Mistral
def answer_question_with_mistral(text, question, model="mistral-small"):
    prompt = f"The following is a research paper:\n\n{text}...\n\nAnswer this question: {question}"
    messages = [{"role": "user", "content": prompt}]
    try:
        response = mistral_client.chat(model=model, messages=messages)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error answering question with Mistral: {str(e)}"

# Answering questions with Gemini
def answer_question_with_gemini(text, question):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(f"The following is a research paper:\n\n{text[:4000]}...\n\nAnswer this question: {question}")
    return response.text.strip() if response and response.text else "Error: No response from Gemini."

# Multi-step question answering for long texts
def answer_long_text(text, question, model_choice):
    chunks = list(chunk_text(text, chunk_size=10000))
    partial_answers = []
    for chunk in chunks:
        if model_choice == "Mistral":
            answer = answer_question_with_mistral(chunk, question)
        elif model_choice == "Gemini":
            answer = answer_question_with_gemini(chunk, question)
        else:
            answer = answer_question_with_cohere(chunk, question)
        partial_answers.append(answer)
    combined_answer = "\n".join(partial_answers)

    final_prompt = (
        f"The user asked: {question}\n\n"
        f"Here are partial answers from different chunks:\n{combined_answer}\n\n"
        "Please provide a final, comprehensive answer to the user's question."
    )
    messages = [{"role": "user", "content": final_prompt}]
    try:
        response = mistral_client.chat(model="mistral-small", messages=messages)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error answering question in final step: {str(e)}"

# Format text with newlines
def format_summary_with_newlines(summary):
    return summary.replace(". ", ".\n")

# CLI for testing
if __name__ == "__main__":
    pdf_path = input("Enter the full path to the PDF file: ").strip()
    try:
        text = extract_text_from_pdf(pdf_path)

        print("\nChoose a model:")
        print("1. Cohere")
        print("2. Mistral")
        print("3. Gemini")
        choice = input("Enter choice (1, 2, or 3): ").strip()

        if choice == "1":
            print("\nSummarizing the paper using Cohere...")
            summary = summarize_long_text(text, "Cohere")
        elif choice == "2":
            print("\nSummarizing the paper using Mistral...")
            summary = summarize_long_text(text, "Mistral")
        else:
            print("\nSummarizing the paper using Gemini...")
            summary = summarize_long_text(text, "Gemini")

        formatted_summary = format_summary_with_newlines(summary)
        print("\nSummary:\n", formatted_summary)

        while True:
            question = input("\nAsk a question about the paper (or type 'exit' to quit): ")
            if question.lower() == "exit":
                break

            # if choice == "1":
            #     answer = answer_long_text(text, question, "Cohere")
            # elif choice == "2":
            #     answer = answer_long_text(text, question, "Mistral")
            # else:
            #     answer = answer_long_text(text, question, "Gemini")
            #
            # print("\nAnswer:\n", answer)
            responses = {
                "Cohere": answer_long_text(text, question, "Cohere"),
                "Mistral": answer_long_text(text, question, "Mistral"),
                "Gemini": answer_long_text(text, question, "Gemini"),
            }

            similarity_scores, differences = verify_llm_responses(question, responses)

            # Print the answers from each LLM
            print("\nLLM Responses:")
            for model, response in responses.items():
                print(f"\n{model}:\n{response}")

            # Print similarity scores
            print("\nSimilarity Scores:")
            for pair, score in similarity_scores.items():
                print(f"{pair}: {score:.4f}")

            # Print highlighted differences
            print("\nDifferences Between Responses:")
            for pair, diff_text in differences.items():
                print(f"\n{pair}:\n{diff_text}\n")


    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
