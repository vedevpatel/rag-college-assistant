import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
from tqdm import tqdm

# LangChain components
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document # Import Document class

# --- Configuration ---
SCRAPED_DATA_PATH = "scraped_data.jsonl"
CHROMA_DB_PATH = "./chroma_db" # Directory to store the vector database
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # Efficient and good quality
CHUNK_SIZE = 1000 # Characters per chunk
CHUNK_OVERLAP = 150 # Characters overlap between chunks
LLM_MODEL_NAME = "gemini-1.5-flash-latest" # Fast and capable Gemini model
TOP_K_RESULTS = 5 # Number of relevant chunks to retrieve

# --- Helper Functions ---

def load_docs_from_jsonl(filepath):
    """Loads documents from a JSON Lines file."""
    docs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading documents"):
            try:
                data = json.loads(line)
                # Create LangChain Document objects
                doc = Document(
                    page_content=data.get("content", ""),
                    metadata={
                        "source": data.get("source", "Unknown Source"),
                        "page_title": data.get("page_title", "Untitled")
                    }
                )
                # Basic check to avoid adding empty documents
                if doc.page_content and len(doc.page_content.split()) > 5:
                    docs.append(doc)
            except json.JSONDecodeError:
                print(f"[WARN] Skipping invalid JSON line in {filepath}")
            except Exception as e:
                 print(f"[WARN] Error processing line: {e}")
    return docs

def setup_vector_store(docs, embedding_model):
    """Chunks documents and creates/loads a Chroma vector store."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    print(f"Splitting {len(docs)} documents into chunks...")
    split_docs = text_splitter.split_documents(docs)
    print(f"Created {len(split_docs)} chunks.")

    if not split_docs:
        raise ValueError("No documents were successfully split into chunks. Check input data.")

    print(f"Initializing Chroma vector store at '{CHROMA_DB_PATH}'...")
    # Use SentenceTransformerEmbeddings directly
    embedding_function = SentenceTransformerEmbeddings(model_name=embedding_model)

    # Check if DB exists, otherwise create it
    if os.path.exists(CHROMA_DB_PATH) and os.listdir(CHROMA_DB_PATH):
         print("Loading existing vector store...")
         vector_store = Chroma(
             persist_directory=CHROMA_DB_PATH,
             embedding_function=embedding_function
         )
         # Optional: Add new documents if needed, handling potential duplicates
         # vector_store.add_documents(split_docs) # Be careful with duplicates if re-running
         print("Existing vector store loaded.")
    else:
        print("Creating new vector store (this may take a while)...")
        vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=embedding_function,
            persist_directory=CHROMA_DB_PATH
        )
        print("Vector store created and persisted.")

    return vector_store

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load API Key for Google Gemini
    load_dotenv() # Loads variables from a .env file in the same directory
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found. Please set it in a .env file.")
    genai.configure(api_key=api_key)

    # 2. Load Scraped Data
    if not os.path.exists(SCRAPED_DATA_PATH):
        raise FileNotFoundError(f"Scraped data file not found: {SCRAPED_DATA_PATH}. Run scraper.py first.")
    documents = load_docs_from_jsonl(SCRAPED_DATA_PATH)

    if not documents:
        print("No documents loaded. Exiting.")
        exit()

    # 3. Setup Vector Store (Create or Load)
    vector_store = setup_vector_store(documents, EMBEDDING_MODEL_NAME)

    # 4. Initialize LLM
    try:
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, temperature=0.2) # Lower temp for more factual answers
    except Exception as e:
        print(f"Error initializing LLM. Check API key and model name: {e}")
        exit()

    # 5. Create Retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K_RESULTS})

    # 6. Create RAG Chain
    # Prompt template tells the LLM how to use the context
    prompt_template = ChatPromptTemplate.from_template(
        """You are an assistant for question-answering tasks for Las Positas College (LPC).
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer from the context, just say that you don't know.
        Use three sentences maximum and keep the answer concise.
        Cite the source URL or page title from the metadata where possible.

        Question: {input}

        Context:
        {context}

        Answer:"""
    )

    # Chain to combine documents into a single string for the LLM context
    document_chain = create_stuff_documents_chain(llm, prompt_template)

    # Chain that takes user input, retrieves documents, then feeds them to the document_chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    print("\n--- Las Positas College Q&A Bot ---")
    print("Enter your question, or type 'quit' to exit.")

    # 7. Question Answering Loop
    while True:
        query = input("\nYour Question: ")
        if query.lower() == 'quit':
            break
        if not query.strip():
            continue

        try:
            print("Searching...")
            # Invoke the chain
            response = retrieval_chain.invoke({"input": query})

            # Print the answer
            print("\nAnswer:")
            print(response["answer"])

            # Print sources (optional but good practice)
            print("\nSources Considered:")
            seen_sources = set()
            for doc in response["context"]:
                 source_info = doc.metadata.get('source', 'N/A')
                 if source_info not in seen_sources:
                    print(f"- {doc.metadata.get('page_title', 'Untitled')}: {source_info}")
                    seen_sources.add(source_info)
                 if len(seen_sources) >= TOP_K_RESULTS: # Avoid too many sources if chunks are from same page
                     break
            if not seen_sources:
                print("- No specific sources retrieved or context was empty.")

        except Exception as e:
            print(f"\n[ERROR] An error occurred while processing your question: {e}")

    print("\nGoodbye!")