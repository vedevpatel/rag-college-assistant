# The LPC Hawk Bot 🦅

A question-answering assistant for Las Positas College (LPC) using Retrieval-Augmented Generation (RAG) based on information from the official LPC website.

**➡️ [Try the Live App Here!](YOUR_DEPLOYED_APP_URL_HERE) ⬅️**
hawkbot.streamlit.app

## Preview

## ✨ Features

* Answers questions about Las Positas College based on website context.
* Uses RAG to provide context-aware answers.
* Retrieves relevant text chunks from scraped website data.
* Attempts to cite sources or context used (details in expander).
* Simple and intuitive Streamlit interface.

## 🛠️ Technology Stack

* **Language:** Python
* **Web Framework:** Streamlit
* **LLM Framework:** LangChain
* **LLM:** Google Generative AI (Gemini Flash)
* **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
* **Vector Store:** ChromaDB

## 🚀 Running Locally (For Developers)

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/vedevpatel/rag-college-assistant.git](https://github.com/vedevpatel/rag-college-assistant.git)
    cd rag-college-assistant
    ```
2.  **Create and activate a virtual environment:** (Recommended)
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set up API Key:**
    * Create a `.env` file in the root directory.
    * Add your Google API key to it:
        ```env
        GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
        ```
5.  **Make Sure the Vector Database is Present:**
    * Make sure the `chroma_db` folder (containing the pre-built vector store) is present in the root directory. If not, you may need to run the original data scraping/indexing script (if you have one).
6.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

## ❓ How to Use (Live App)

1.  Visit https://hawkbot.streamlit.app/
2.  Wait for the resources to load ("Loading resources..." -> "Ready!").
3.  Type your question about Las Positas College into the input box.
4.  Press Enter or click outside the box.
5.  View the generated answer.
6.  Optionally, expand the "Show Sources Considered" section to see the context used.

## ⚙️ Configuration

* **Google API Key:** A `GOOGLE_API_KEY` with the Generative Language API enabled is required. Provide this via a `.env` file locally or as a Secret (named `GOOGLE_API_KEY`) during deployment (e.g., on Streamlit Community Cloud).
* **(If Feedback Feature Added):** Email credentials (`EMAIL_USER`, `EMAIL_PASS`) may be needed as environment variables/secrets if the email feedback feature is implemented.

## 📧 Feedback and Bug Reports

Found an issue or have suggestions? Please feel free to reach out directly via email at: `vedevpatel@gmail.com`
