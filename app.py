import os
import PyPDF2
import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

# Initialize global variables
vector_db = None
llm = None
retriever = None
chain = None

def process_pdf(file):
    global vector_db, llm, retriever, chain
    
    # Extract text from the PDF
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    
    # Convert the extracted text into a Document object
    document = Document(page_content=text)
    
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    
    # Split the document into chunks
    chunks = text_splitter.split_documents([document])
    
    # Initialize the vector database with the split chunks
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
        collection_name="local-rag"
    )
    
    # Define the LLM from Ollama
    local_model = "llama3"
    global llm
    llm = ChatOllama(model=local_model)
    
    # Define the prompt template for generating multiple queries
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )
    
    # Initialize the retriever
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), 
        llm,
        prompt=QUERY_PROMPT
    )
    
    # Define the RAG prompt template
    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """
    
    # Initialize the chain
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

def main():
      
    st.title("PDF ChatBot")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        st.write("Processing your PDF...")
        process_pdf(uploaded_file)
        st.write("PDF processing complete.")
        
        question = st.text_input("Enter your question about the PDF:")
        
        if st.button("Get Answer"):
            if question and chain:
                response = chain.invoke(question)
                st.write("Response:", response)
            else:
                st.write("Please upload a PDF and enter a question.")
    
    if st.button("Cleanup"):
        global vector_db
        if vector_db:
            vector_db.delete_collection()
            st.write("Vector database cleaned up.")

if __name__ == "__main__":
    main()

