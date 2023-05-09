from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
# from langchain.llms import OpenAI
# from langchain.callbacks import get_openai_callback


def main():
    load_dotenv()
    st.set_page_config(page_title='PDF query tool')
    st.header('Query the PDF')
    pdf = st.file_uploader("Upload source PDF here", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        text_splitter = CharacterTextSplitter(
            separator='\n',
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text)

        embeddings = HuggingFaceEmbeddings()
        knowledgebase = FAISS.from_texts(chunks, embeddings)

        st.write('Embeddings generated')

        # initialize huggingFace LLM
        flan_t5 = HuggingFaceHub(
            repo_id = 'google/flan-t5-base',
            model_kwargs = {'temperature': 0}
        )

        user_query = st.text_input("input your query here:")
        if user_query:
            docs = knowledgebase.similarity_search(user_query)
            
            # llm = OpenAI(temperature=0)
            llm = flan_t5
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents = docs, question = user_query)
            # with get_openai_callback() as cb:
            #     response = chain.run(input_documents = docs, question = user_query)
            #     print(cb)
            st.write(response)


if __name__== '__main__':
    main()
