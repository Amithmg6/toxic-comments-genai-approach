import faiss
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint # Updated import
from langchain_ollama import OllamaLLM # Updated import
from dotenv import load_dotenv

load_dotenv()

def build_vector_database(data_path="data/toxic_comments_dataset.csv", index_path="models/faiss_index.bin"):
    """ Loading data, create embbedings, and builds faiss index.""" 
    print("Build the vector database...")

    df = pd.read_csv(data_path)

    ## initialize embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    ## generate embeddings
    embeddings = model.encode(df['comment'].to_list())

    ## build faiss index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))

    ## save the index
    faiss.write_index(index, index_path)
    df.to_parquet("models/comments_data.parquet")

    print("Vector database built and saved.")

## step 2 inference phase

def classify_comment_with_rag(new_comment, k=5, index_path="models/faiss_index.bin", data_path="models/comments_data.parquet"):
    """ Classify a new comment using RAG approach."""
    print("Classifying new comment using a RAG based system...")

    ## load data and index
    try:
        index = faiss.read_index(index_path)
        comments_df = pd.read_parquet("models/comments_data.parquet")
    except FileNotFoundError:
        return "Error: vector database not found. Execute indexing phase first."

    # initialize same embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # generate the embedding for the new comment
    new_comment_embedding = model.encode([new_comment])

    # perform a similarity search
    D, I = index.search(np.array(new_comment_embedding).astype('float32'), k)

    # retrival of the top 5 comments
    retrieved_context = comments_df.iloc[I[0]]

    # format the retrieved examples for the llm prompt
    context_str = ""
    for _, row in retrieved_context.iterrows():
        context_str += f"comment: '{row['comment']}' | Label: {row['label']}\n"

    # define the prompt template
    template = """
    You are an expert toxic comment classifier. You will classify a new comment as 'toxic' or 'not toxic'.
    
    Here are some examples of comments and their labels to help you classify the new comment:
    
    {context}
    
    Based on the examples above, please classify the following comment:
    
    Comment: '{new_comment}'
    
    Classification:
    """

    prompt = PromptTemplate(template=template, input_variables=["context", "new_comment"])

    ## using hugging face model for llm
    
    llm = OllamaLLM(model="llama3") #, temperature=0.1, max_new_tokens=50)

    # llm = HuggingFaceEndpoint(
    #     repo_id="distilbert-base-uncased-finetuned-sst-2-english",
    #     task="text-classification",
    #     temperature=0.1,
    #     max_new_tokens=50,
    #     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    # )

    # generate the full prompt and get llm response
    full_prompt = prompt.format(context=context_str, new_comment=new_comment)
    llm_response = llm.invoke(full_prompt)

# Example usage of the classifier logic
if __name__ == "__main__":
    # Create a dummy CSV file for demonstration
    dummy_data = {
        'comment': [
            "You are a terrible person.", 
            "I love this movie, it's so great!", 
            "The weather is horrible today.", 
            "You are an idiot.", 
            "This is a great day!", 
            "What an amazing project.",
            "You look ugly.",
            "The traffic is very bad."
        ],
        'label': [
            'toxic', 
            'not toxic', 
            'not toxic', 
            'toxic', 
            'not toxic',
            'not toxic',
            'toxic',
            'not toxic'
        ]
    }
    dummy_df = pd.DataFrame(dummy_data)
    
    # Ensure the directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    dummy_df.to_csv("data/toxic_comments_dataset.csv", index=False)
    
    # Run the indexing phase
    build_vector_database()
    
    # Now, test the inference phase
    test_comment = "I hate you, you are worthless."
    print(f"\nClassifying: '{test_comment}'")
    classification = classify_comment_with_rag(test_comment)
    print(f"Classification Result: {classification}")