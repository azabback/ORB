import os
import fitz  # PyMuPDF for PDF reading
import cohere
from mistralai.client import MistralClient




from dotenv import load_dotenv


load_dotenv()


COHERE_API_KEY = os.getenv("COHERE_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not COHERE_API_KEY or not MISTRAL_API_KEY:
    raise ValueError("Missing API keys. Please check your .env file.")


cohere_client = cohere.Client(COHERE_API_KEY)

mistral_client = MistralClient(api_key=MISTRAL_API_KEY)


def extract_text_from_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Error: PDF file not found at '{pdf_path}'")

    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text

def summarize_with_cohere(text, model="command-r-plus"):
    prompt = f"Summarize the following research paper:\n\n{text[:3000]}..."
    response = cohere_client.generate(
        model=model,
        prompt=prompt,
        max_tokens=300
    )
    return response.generations[0].text.strip()

def answer_question_with_cohere(text, question, model="command-r-plus"):
    prompt = f"The following is a research paper:\n\n{text[:4000]}...\n\nAnswer this question: {question}"
    response = cohere_client.generate(
        model=model,
        prompt=prompt,
        max_tokens=300
    )
    return response.generations[0].text.strip()


def chunk_text(text, chunk_size=10000):
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]


def summarize_with_mistral(text, model="mistral-small"):
    shortened_text = text[:15000]  #
    prompt = f"Summarize the following research paper:\n\n{shortened_text}..."
    messages = [{"role": "user", "content": prompt}]
    try:
        response = mistral_client.chat(model=model, messages=messages)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error summarizing with Mistral: {str(e)}"


def summarize_long_text(text):

    chunks = list(chunk_text(text, chunk_size=10000))
    partial_summaries = []

    for chunk in chunks:
        summary = summarize_with_mistral(chunk)
        partial_summaries.append(summary)

    combined_text = "\n".join(partial_summaries)
    final_summary = summarize_with_mistral(combined_text)
    return final_summary


def summarize_with_mistral(text, model="mistral-small"):
    prompt = f"Summarize the following research paper:\n\n{text}..."
    messages = [{"role": "user", "content": prompt}]
    try:
        response = mistral_client.chat(model=model, messages=messages)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error summarizing with Mistral: {str(e)}"


# def answer_question_with_mistral(text, question, model="mistral-small"):
#     """用 Mistral API 进行问答"""
#     prompt = f"The following is a research paper:\n\n{text}...\n\nAnswer this question: {question}"
#     messages = [{"role": "user", "content": prompt}]
#
#     try:
#         response = mistral_client.chat(model=model, messages=messages)
#         return response.choices[0].message.content.strip()
#     except Exception as e:
#         return f"Error answering question with Mistral: {str(e)}"


def answer_long_text_mistral(text, question, model="mistral-small", chunk_size=10000):
    chunks = list(chunk_text(text, chunk_size=chunk_size))

    partial_answers = []

    for i, chunk in enumerate(chunks):
        print(f"[DEBUG] Processing chunk {i + 1}/{len(chunks)} for QA...")
        partial_answer = answer_question_with_mistral(chunk, question, model=model)
        partial_answers.append(partial_answer)

    combined_partial = "\n".join(partial_answers)

    final_prompt = (
        f"The user asked: {question}\n\n"
        f"Here are partial answers from different chunks:\n{combined_partial}\n\n"
        "Please provide a final, comprehensive answer to the user's question."
    )
    messages = [{"role": "user", "content": final_prompt}]

    try:
        response = mistral_client.chat(model=model, messages=messages)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error answering question with Mistral (final step): {str(e)}"


def answer_question_with_mistral(text, question, model="mistral-small"):
    prompt = f"The following is a research paper:\n\n{text}...\n\nAnswer this question: {question}"
    messages = [{"role": "user", "content": prompt}]
    try:
        response = mistral_client.chat(model=model, messages=messages)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error answering question with Mistral: {str(e)}"

def format_summary_with_newlines(summary):
    return summary.replace(". ", ".\n")

if __name__ == "__main__":
    pdf_path = input("Enter the full path to the PDF file: ").strip()

    try:
        text = extract_text_from_pdf(pdf_path)

        print("\nChoose a model:")
        print("1. Cohere")
        print("2. Mistral")
        choice = input("Enter choice (1 or 2): ").strip()

        if choice == "1":
            print("\nSummarizing the paper using Cohere...")
            summary = summarize_with_cohere(text)
        else:
            print("\nSummarizing the paper using Mistral...")
            summary = summarize_long_text(text)

        formatted_summary = format_summary_with_newlines(summary)
        print("\nSummary:\n", formatted_summary)

        while True:
            question = input("\nAsk a question about the paper (or type 'exit' to quit): ")
            if question.lower() == "exit":
                break

            if choice == "1":
                answer = answer_question_with_cohere(text, question)
            else:
                # answer = answer_question_with_mistral(text, question)
                answer = answer_long_text_mistral(text, question)

            print("\nAnswer:\n", answer)

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
