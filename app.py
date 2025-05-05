__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
import time # Keep track of time for potential debugging

# LangChain Imports (ensure these match your installed versions)
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# --- Configuration ---
CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # Make sure this matches what you used to build the DB
LLM_MODEL_NAME = "gemini-1.5-flash-latest"
TOP_K_RESULTS = 5 # Number of relevant chunks to retrieve

# --- Caching Functions (Crucial for Performance) ---
# These ensure slow operations only run once per session or when code changes

# Cache the embedding function resource across sessions
@st.cache_resource
def get_embedding_function():
    st.write(f"Loading embedding function ({EMBEDDING_MODEL_NAME})...") # Use st.write for visibility in app
    start_time = time.time()
    # Ignore deprecation warnings for now, adjust imports later if needed
    # e.g., from langchain_huggingface import HuggingFaceEmbeddings
    try:
         embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
         end_time = time.time()
         st.write(f"Embedding function loaded in {end_time - start_time:.2f} seconds.")
         return embedding_function
    except Exception as e:
         st.error(f"Failed to load embedding function: {e}")
         return None

# Cache the LLM resource across sessions
@st.cache_resource
def load_llm():
    st.write(f"Loading LLM ({LLM_MODEL_NAME})...")
    start_time = time.time()
    # Load API key from .env file OR Streamlit secrets
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY") # Checks .env locally, checks secrets when deployed

    if not api_key:
        st.error("🔴 GOOGLE_API_KEY not found! Set it in .env file locally, or in Streamlit Secrets when deployed.")
        return None
    try:
        # Configure GenAI (important step)
        genai.configure(api_key=api_key)
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, temperature=0.2)
        end_time = time.time()
        st.write(f"LLM loaded in {end_time - start_time:.2f} seconds.")
        return llm
    except Exception as e:
        st.error(f"🔴 Failed to initialize LLM: {e}")
        st.error("Ensure your API key is correct and has the Generative Language API enabled.")
        return None

# Cache the vector store resource across sessions
@st.cache_resource
def load_vector_store(_embedding_func): # Pass embedding func as argument
     st.write(f"Loading vector store from '{CHROMA_DB_PATH}'...")
     start_time = time.time()
     if not os.path.exists(CHROMA_DB_PATH) or not os.listdir(CHROMA_DB_PATH):
          st.error(f"🔴 Chroma DB not found at '{CHROMA_DB_PATH}'.")
          st.error("Please ensure the 'chroma_db' folder exists and contains data.")
          st.error("Run the initial scraping and indexing script if needed.")
          return None
     if _embedding_func is None:
          st.error("🔴 Cannot load vector store without a valid embedding function.")
          return None
     try:
        # Ignore deprecation warnings for now, adjust imports later if needed
        # e.g., from langchain_chroma import Chroma
        vector_store = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=_embedding_func # Use the passed function
        )
        end_time = time.time()
        st.write(f"Vector store loaded in {end_time - start_time:.2f} seconds.")
        return vector_store
     except Exception as e:
          st.error(f"🔴 Failed to load vector store: {e}")
          return None

# --- Main App Logic ---

st.set_page_config(layout="wide") # Use wider layout
st.title("🦅The LPC Hawk Bot🦅")
st.caption("Ask questions about LPC based on information from its website.")
st.divider()

# --- Load resources early and show status ---
with st.status("Initializing resources...", expanded=False) as status:
    embedding_function = get_embedding_function()
    llm = load_llm()
    vector_store = load_vector_store(embedding_function)
    status.update(label="Initialization complete!", state="complete")


if llm and vector_store and embedding_function:
    # --- Create RAG Chain (This part is fast, no need to cache typically) ---
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K_RESULTS})
    prompt_template = ChatPromptTemplate.from_template(
        """You are an assistant for question-answering tasks for Las Positas College (LPC).
        Use ONLY the following pieces of retrieved context to answer the question.
        If you don't know the answer from the context, explicitly state that you don't know based on the provided documents.
        Do not add information that is not present in the context.
        Keep the answer concise and helpful, ideally within 3-4 sentences.
        If possible, mention the source document title or URL found in the metadata of the context.

        Question: {input}

        Context:
        {context}

        Answer:"""
    )
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # --- User Interface ---
    st.subheader("Ask your question:")
    query = st.text_input("Enter question here:", key="query_input", label_visibility="collapsed")

    if query:
        with st.spinner("Searching and generating answer..."):
            try:
                start_query_time = time.time()
                response = retrieval_chain.invoke({"input": query})
                end_query_time = time.time()

                st.info(f"Response generated in {end_query_time - start_query_time:.2f} seconds.")

                st.subheader("Answer:")
                st.markdown(response["answer"]) # Use markdown for better formatting

                # Display sources toggle
                with st.expander("Show Sources Considered"):
                    seen_sources = set()
                    sources_text = ""
                    if "context" in response and response["context"]:
                         sources_text += f"*Retrieved {len(response['context'])} context chunks:*\n\n"
                         for i, doc in enumerate(response["context"]):
                              source_info = doc.metadata.get('source', 'N/A')
                              page_title = doc.metadata.get('page_title', 'Untitled')
                              source_id = f"{page_title} ({source_info})"
                              # Show each source chunk for debugging/transparency
                              sources_text += f"**Chunk {i+1}: {page_title}** ({source_info})\n"
                              sources_text += f"> {doc.page_content[:300]}...\n\n" # Show beginning of chunk
                              seen_sources.add(source_id) # Add unique source identifier

                    if not seen_sources:
                         sources_text = "No specific sources retrieved or context was empty."
                    st.markdown(sources_text)


            except Exception as e:
                st.error(f"An error occurred while processing your query: {e}")
                st.exception(e) # Print full traceback for debugging

else:
    st.error("🔴 Bot could not be initialized. Please check errors in the status area above or the console output.")
    st.warning("Ensure your API key is valid and the Chroma database exists.")

# Add a footer or sidebar info if desired
st.sidebar.header("About")
st.sidebar.info("This chatbot uses information scraped from the Las Positas College website. Its knowledge is limited to the college website.")