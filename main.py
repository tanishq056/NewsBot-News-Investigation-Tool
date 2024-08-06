import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fetch the API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OPENAI_API_KEY not found in environment variables")
    st.stop()  # Stop the Streamlit app if the API key is not found

# Initialize OpenAI with the API key
llm = OpenAI(openai_api_key=openai_api_key, temperature=0.9, max_tokens=500)

st.title("News Analyzer Tool")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()

if process_url_clicked:
    # Load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    # Split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    print(f"Number of documents: {len(docs)}")

    if not docs:
        st.error("No documents were processed.")
    else:
        embeddings = OpenAIEmbeddings()
        try:
            sample_embedding = embeddings.embed_documents([docs[0]])
            print("Sample embedding dimension:", len(sample_embedding[0]))
            vectorstore_openai = FAISS.from_documents(docs, embeddings)
            main_placeholder.text("Embedding Vector Started Building...✅✅✅")
            time.sleep(2)

            # Save the FAISS index to a pickle file
            with open(file_path, "wb") as f:
                pickle.dump(vectorstore_openai, f)

        except Exception as e:
            st.error(f"Error creating FAISS index: {e}")

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)
