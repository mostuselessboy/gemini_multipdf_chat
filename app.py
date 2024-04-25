import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
import random
# load_dotenv()
# os.getenv("GOOGLE_API_KEY")
genai.configure(api_key="AIzaSyAuo39Tdn6eWUYBcpXhM3LRTn67ycVqbx0")

# read all pdf files and return text
def get_random_sample():
    sample_inputs = [
        "What are the legal provisions related to cybercrimes?",
        "How can I report a crime in Delhi?",
        "What are the do's and don'ts during an emergency?",
        "Tell me about the South District of Delhi.",
        "How does the Delhi Police handle cases of defacement?",
        "What is forced deployment in the context of Delhi Police?",
        "Tell me about the latest e-campaign by Delhi Police.",
        "Explain the concept of 'Glance' in the Delhi Police context."
    ]
    return random.choice(sample_inputs)
    sample_inputs = [
        "What are the legal provisions related to cybercrimes?",
        "How can I report a crime in Delhi?",
        "What are the do's and don'ts during an emergency?",
        "Tell me about the South District of Delhi.",
        "How does the Delhi Police handle cases of defacement?",
        "What is forced deployment in the context of Delhi Police?",
        "Tell me about the latest e-campaign by Delhi Police.",
        "Explain the concept of 'Glance' in the Delhi Police context."
    ]

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# split text into chunks


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks  # list of strings

# get embeddings for each chunk


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
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain


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
        response = chain.invoke(
            {"input_documents": docs, "question": user_question}, return_only_outputs=True, )
    except Exception:
        return {'output_text':["AI Cannot Answer these type of Questions for Safety Reason"]}
    print(response)
    return response


def main():
    st.set_page_config(
        page_title="Delhi Police Bot",
        page_icon="🤖"
    )


    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                .viewerBadge_link__qRIco{display:none;}
                </style>
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)




    # Main content area for displaying chat messages
    # st.image("botheader.png", caption="", use_column_width=True)

    st.title("👮Delhi Police ChatBot💬")
    st.write("दिल्ली पुलिस आपकी सेवा में 🙏")
    st.button('Clear Chat History', on_click=clear_chat_history)

    # Initialize chat history if not already present
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Delhi Police Seva mein aapka swagat hai 🙏"}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    col1, col2 = st.columns(2)
    with col1:
        if st.button(sample_questions[0]):
            st.session_state.messages.append({"role": "user", "content": sample_questions[0]})
            with st.chat_message("user"):
                st.write(sample_questions[0])

    with col2:
        if st.button(sample_questions[1]):
            st.session_state.messages.append({"role": "user", "content": sample_questions[1]})
            with st.chat_message("user"):
                st.write(sample_questions[1])

    col3, col4 = st.columns(2)
    with col3:
        if st.button(sample_questions[2]):
            st.session_state.messages.append({"role": "user", "content": sample_questions[2]})
            with st.chat_message("user"):
                st.write(sample_questions[2])

    with col4:
        if st.button(sample_questions[3]):
            st.session_state.messages.append({"role": "user", "content": sample_questions[3]})
            with st.chat_message("user"):
                st.write(sample_questions[3])

    col5, col6 = st.columns(2)
    with col5:
        if st.button(sample_questions[4]):
            st.session_state.messages.append({"role": "user", "content": sample_questions[4]})
            with st.chat_message("user"):
                st.write(sample_questions[4])

    with col6:
        if st.button(sample_questions[5]):
            st.session_state.messages.append({"role": "user", "content": sample_questions[5]})
            with st.chat_message("user"):
                st.write(sample_questions[5])

    col7, col8 = st.columns(2)
    with col7:
        if st.button(sample_questions[6]):
            st.session_state.messages.append({"role": "user", "content": sample_questions[6]})
            with st.chat_message("user"):
                st.write(sample_questions[6])

    with col8:
        if st.button(sample_questions[7]):
            st.session_state.messages.append({"role": "user", "content": sample_questions[7]})
            with st.chat_message("user"):
                st.write(sample_questions[7])
                
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Display bot response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                pdf_docs = ["glance.pdf","southdistricteng.pdf","ecampaign.pdf", "legalprovision.pdf", "doanddont.pdf", "forcedepl.pdf", "defacement.pdf"]
                # with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                response = user_input(prompt)
                full_response = ''
                for item in response['output_text']:
                    full_response += item
                st.write(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)


if __name__ == "__main__":
    main()

