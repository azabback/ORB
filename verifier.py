from sentence_transformers import SentenceTransformer, util
import difflib

# Load a sentence embedding model for similarity comparison
embedder = SentenceTransformer('all-MiniLM-L6-v2')


def compute_similarity(response1, response2):
    """
    Compute cosine similarity between two LLM responses.
    """
    embeddings = embedder.encode([response1, response2], convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return similarity.item()  # Convert tensor to float


def compare_llm_responses(responses):
    """
    Compare multiple LLM responses and compute similarity scores.
    responses: dict with model names as keys and response texts as values.
    """
    model_names = list(responses.keys())
    scores = {}

    # Compare each response pairwise
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            model1, model2 = model_names[i], model_names[j]
            score = compute_similarity(responses[model1], responses[model2])
            scores[f"{model1} vs {model2}"] = round(score, 4)

    return scores


def highlight_differences(text1, text2):
    """
    Highlight differences between two LLM responses.
    """
    diff = difflib.ndiff(text1.split(), text2.split())
    diff_output = ' '.join([word if word[0] == ' ' else f"**{word}**" for word in diff])
    return diff_output


def verify_llm_responses(question, responses):
    """
    Main function to compare LLM responses, compute similarity scores, and highlight differences.
    """
    similarity_scores = compare_llm_responses(responses)

    differences = {}
    for pair in similarity_scores.keys():
        model1, model2 = pair.split(" vs ")
        differences[pair] = highlight_differences(responses[model1], responses[model2])

    return similarity_scores, differences
