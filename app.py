import os
import fitz  # PyMuPDF for PDF reading
import gradio as gr
import cohere
from mistralai.client import MistralClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Retrieve API Keys
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not COHERE_API_KEY or not MISTRAL_API_KEY:
    raise ValueError("Missing API keys. Please check your .env file.")

# Initialize API clients
cohere_client = cohere.Client(COHERE_API_KEY)
mistral_client = MistralClient(api_key=MISTRAL_API_KEY)


# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text("text") for page in doc])


# Split long text into smaller chunks
def chunk_text(text, chunk_size=10000):
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]


# Generate summary based on the selected model
def summarize_text(text, model_choice):
    return summarize_with_cohere(text) if model_choice == "Cohere" else summarize_long_text(text)


# Cohere summary
def summarize_with_cohere(text, model="command-r-plus"):
    prompt = f"Summarize the following research paper:\n\n{text[:3000]}..."
    response = cohere_client.generate(
        model=model,
        prompt=prompt,
        max_tokens=300
    )
    return response.generations[0].text.strip()


# Mistral summary with automatic chunking
def summarize_long_text(text, model="mistral-small"):
    chunks = list(chunk_text(text, chunk_size=10000))
    partial_summaries = [summarize_with_mistral(chunk, model=model) for chunk in chunks]
    combined_text = "\n".join(partial_summaries)
    return summarize_with_mistral(combined_text, model=model)


def summarize_with_mistral(text, model="mistral-small"):
    prompt = f"Summarize the following research paper:\n\n{text[:15000]}..."
    messages = [{"role": "user", "content": prompt}]

    try:
        response = mistral_client.chat(model=model, messages=messages)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error summarizing with Mistral: {str(e)}"


# Answer user questions using the selected model
def answer_question(text, question, model_choice):
    return answer_question_with_cohere(text, question) if model_choice == "Cohere" else answer_long_text_mistral(text,
                                                                                                                 question)


# Cohere Q&A
def answer_question_with_cohere(text, question, model="command-r-plus"):
    prompt = f"The following is a research paper:\n\n{text[:4000]}...\n\nAnswer this question: {question}"
    response = cohere_client.generate(
        model=model,
        prompt=prompt,
        max_tokens=300
    )
    return response.generations[0].text.strip()


# Mistral Q&A with automatic chunking
def answer_long_text_mistral(text, question, model="mistral-small", chunk_size=10000):
    chunks = list(chunk_text(text, chunk_size=chunk_size))
    partial_answers = [answer_question_with_mistral(chunk, question, model=model) for chunk in chunks]
    combined_partial = "\n".join(partial_answers)

    final_prompt = (
        f"The user asked: {question}\n\n"
        f"Here are partial answers from different chunks:\n{combined_partial}\n\n"
        "Please provide a final, comprehensive answer."
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


# Process file and generate summary
def process_file(file, model_choice):
    pdf_path = file
    text = extract_text_from_pdf(pdf_path)
    summary = summarize_text(text, model_choice)
    return text, summary


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 📄 PDF Summarizer & QA")

    with gr.Row():
        file_input = gr.File(label="Upload a PDF", type="filepath")
        model_choice = gr.Radio(["Cohere", "Mistral"], label="Choose a Model", value="Cohere")

    with gr.Row():
        process_button = gr.Button("Summarize")

    with gr.Row():
        text_output = gr.Textbox(label="Extracted Text", interactive=False)
        summary_output = gr.Textbox(label="Summary", interactive=False)

    process_button.click(process_file, inputs=[file_input, model_choice], outputs=[text_output, summary_output])

    gr.Markdown("## 🔍 Ask a Question about the Paper")

    with gr.Row():
        question_input = gr.Textbox(label="Your Question")
        question_button = gr.Button("Get Answer")

    with gr.Row():
        answer_output = gr.Textbox(label="Answer", interactive=False)

    question_button.click(answer_question, inputs=[text_output, question_input, model_choice], outputs=answer_output)

# Launch Gradio app
demo.launch()
