import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import streamlit as st
import google.generativeai as genai
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

# Initialize GenAI
genai.configure(api_key="AIzaSyAuo39Tdn6eWUYBcpXhM3LRTn67ycVqbx0")

def read_pdf(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    return ' '.join([page.extract_text() for page in pdf_reader.pages])

def get_pdf_text(pdf_docs):
    return ' '.join([read_pdf(pdf) for pdf in pdf_docs])

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question from the context in points by rephrasing it. GIVE ANSWER ONLY IN POINTS if the answer is not in
    provided context just say, "Please be more concise with your question🙏", don't provide the wrong answer instead give them a similar question that can be found in the Context! PREFER THAT ANSWER IN CONTEXT THAT COMES FIRST!\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   client=genai,
                                   temperature=0.8, 
                                   safety_settings={
                                       genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                                       genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                                       genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                                       genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE
                                   }
                                   )
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)

def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "upload some pdfs and ask me a question"}]

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) 
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    try:
        return chain.invoke(
            {"input_documents": docs, "question": user_question}, return_only_outputs=True)
    except Exception:
        return {'output_text':["AI Cannot Answer these type of Questions for Safety Reason"]}

def main():
    st.set_page_config(
        page_title="Delhi Police Bot",
        page_icon="🤖"
    )

    st.markdown("""
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                .viewerBadge_link__qRIco{display:none;}
                </style>
                """, unsafe_allow_html=True)

    st.title("👮Delhi Police ChatBot💬")
    st.write("दिल्ली पुलिस आपकी सेवा में 🙏")
    st.button('Clear Chat History', on_click=clear_chat_history)

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Delhi Police Seva mein aapka swagat hai 🙏"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                pdf_docs = ["glance.pdf","southdistricteng.pdf","ecampaign.pdf", "legalprovision.pdf", "doanddont.pdf", "forcedepl.pdf", "defacement.pdf"]
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                response = user_input(prompt)
                full_response = ''.join(response['output_text'])
                st.write(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
