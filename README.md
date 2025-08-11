# PDF_Chat_Rag_App
#  Chat With Your PDF(s)

This is a Streamlit application that allows you to chat with your PDF documents! Upload one or multiple PDF files, and the AI assistant will answer your questions based on the content of those documents, maintaining conversational context.

##  Features

* **PDF Upload**: Easily upload single or multiple PDF documents.
* **Intelligent Q&A**: Ask questions and get answers directly from the content of your uploaded PDFs.
* **Conversational Memory**: The AI remembers previous turns in the conversation, allowing for natural follow-up questions.
* **Context-Aware Retrieval**: Utilizes a history-aware retriever to reformulate questions based on chat history for better document retrieval.
* **Groq Integration**: Leverages Groq's fast inference for quick AI responses.
* **OpenAI Embeddings**: Uses OpenAI's `text-embedding-3-small` for efficient text vectorization.

##  Setup and Installation

Follow these steps to get your local copy up and running.

### Prerequisites

Before you begin, ensure you have the following:

* **Python 3.8+**
* **Git** (for cloning the repository)
* **API Keys**:
    * **OpenAI API Key**: Required for text embeddings. You can get one from [OpenAI](https://platform.openai.com/account/api-keys).
    * **Groq API Key**: Required for the language model. You can get one from [Groq](https://console.groq.com/keys).

### 1. Clone the Repository

First, clone this GitHub repository to your local machine:

```bash
git clone https://github.com/pavan29903/PDF_Chat_Rag_App.git
```
### 2. Set Up Virtual Environment
```bash
python -m venv myenv
# On Windows:
myenv\Scripts\activate
# On macOS/Linux:
source myenv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys
```bash
OPENAI_API_KEY="your_openai_api_key_here"
GROQ_API_KEY="your_groq_api_key_here"
```

### 5. Run the Application
```bash
streamlit run app.py
```