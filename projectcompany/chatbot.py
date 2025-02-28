import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import spacy


try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Spacy model 'en_core_web_sm' not found. Please install it by running: python -m spacy download en_core_web_sm")

# Define the URLs for the documentation
documentation_urls = {
    "segment": "https://segment.com/docs/?ref=nav",
    "mparticle": "https://docs.mparticle.com/",
    "lytics": "https://docs.lytics.com/",
    "zeotap": "https://docs.zeotap.com/home/en-us/"
}

# Load a pre-trained transformer-based model for question answering
qa_pipeline = pipeline("question-answering")


def fetch_documentation(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        print(f"Successfully fetched the page: {url}")
        page_content = response.content
        soup = BeautifulSoup(page_content, "html.parser")
        return soup
    else:
        print(f"Failed to fetch the page: {url} with status code {response.status_code}")
        return None

# Function to extract text content from the documentation page
def extract_text(soup):
    # You can target specific sections that hold the important documentation
    # For example, extract from <div>, <section>, or <article> tags
    paragraphs = soup.find_all(['p', 'div', 'section', 'article'])
    text = " ".join([para.get_text() for para in paragraphs if para.get_text().strip() != ''])
    return text

# Function to answer questions based on the documentation
def answer_question_from_docs(question, docs_text):
    # Use the question-answering pipeline (transformer-based model)
    answer = qa_pipeline({
        'question': question,
        'context': docs_text
    })
    return answer['answer']

# Function to handle cross-CDP comparison questions
def compare_cdps(question):
    if "segment" in question and "lytics" in question:
        # Extract the relevant sections for Segment and Lytics, and compare them
        segment_soup = fetch_documentation(documentation_urls["segment"])
        lytics_soup = fetch_documentation(documentation_urls["lytics"])
        if segment_soup and lytics_soup:
            segment_text = extract_text(segment_soup)
            lytics_text = extract_text(lytics_soup)
            answer_segment = answer_question_from_docs(question, segment_text)
            answer_lytics = answer_question_from_docs(question, lytics_text)
            return f"Segment: {answer_segment}\nLytics: {answer_lytics}"
    else:
        return "I cannot compare those platforms at the moment."

# Function to process user input and give appropriate answers
def process_user_input(question):
    # Handle variations in questions and potential errors
    if any(keyword in question.lower() for keyword in ["movie", "release", "film"]):
        return "I can only answer questions related to Customer Data Platforms (CDPs). Please ask about Segment, mParticle, Lytics, or Zeotap."

    print(f"Received question: {question}")  # Debugging line

    # Tokenize the question to determine which CDP the question is about
    doc = nlp(question)
    platforms_mentioned = [platform for platform in documentation_urls.keys() if platform in question.lower()]

    if not platforms_mentioned:
        return "Please specify one of the following platforms: Segment, mParticle, Lytics, or Zeotap."

    # Choose a platform and extract relevant documentation
    platform = platforms_mentioned[0]  # Take the first platform mentioned
    platform_soup = fetch_documentation(documentation_urls[platform])

    if platform_soup:
        platform_text = extract_text(platform_soup)
        return answer_question_from_docs(question, platform_text)
    else:
        return f"Could not retrieve documentation for {platform}."

# Example Usage
if __name__ == "__main__":
    while True:
        question = input("Ask me a how-to question (or type 'exit' to quit): ")

        if question.lower() == 'exit':
            break

        if "compare" in question.lower():
            print(compare_cdps(question))
        else:
            print(process_user_input(question))
