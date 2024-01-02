# ------------------------------------------------------ LIBRARIES ------------------------------------------------------ #
# Import base streamlit dependency
import streamlit as st
from streamlit.web import cli as stcli
from streamlit_option_menu import option_menu  # pip install streamlit-option-menu
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space

# Import LangChain dependencies
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import AzureChatOpenAI
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)

# Import system
import sys
import os

# Import other dependencies
from htmlTemplates import css, bot_template, user_template

# Import env variables
from dotenv import load_dotenv

# ------------------------------------------------------ FUNCTIONS ------------------------------------------------------ #

def submit():
  st.session_state.user_input_prompt = st.session_state.user_input
  st.session_state.user_input = ''

def print_file_uploaded_chat():
    with st.spinner("Thinking..."):
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def print_no_file_uploaded_chat(extra_class):
    with st.spinner("Thinking..."):
        no_file_msgs = st.session_state.get('no_file_conversation', [])
        for i, msg in enumerate(no_file_msgs[1:]):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", msg.content).replace("{{EXTRA_CLASS}}", extra_class), unsafe_allow_html=True)

# Function to handle the user input
def handle_user_input(user_question, warning_message=None, extra_class=None):

    if warning_message is None: # Use to handle users that start querying with uploaded documents
        if user_question == 'None':
            print_file_uploaded_chat()
        else:
            response = st.session_state.conversation({'question': user_question})
            st.session_state.chat_history = response['chat_history']
            print_file_uploaded_chat()
    
    else: # Use to handle users that start querying without uploading any documents
        if user_question == 'None':
            print_no_file_uploaded_chat(extra_class)
        else:
            st.session_state.no_file_conversation.append(HumanMessage(content=user_question))
            st.session_state.no_file_conversation.append(AIMessage(content=warning_message))
            print_no_file_uploaded_chat(extra_class)
            
# Function to save the uploaded file to the local Upload folder
def save_uploadedfile(uploadedfile):

    with open(os.path.join("Upload", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())

    # return st.success(body=f"File uploaded!", icon="✅")

# Function to get the text chunks from the PDFs
def get_pdf_text_chunks():

    # Initialize the text chunks list
    text_chunks = []

    # Retrive all the PDF files from the Upload folder. Output of file_list = ['file1.pdf', 'file2.pdf']
    files = filter(lambda f: f.lower().endswith(".pdf"), os.listdir("Upload"))
    file_list = list(files)

    # Loop through the PDF files and extract the text chunks
    for file in file_list:
        
        # Retrieve the PDF file
        loader = PyPDFLoader(os.path.join('Upload', file)) # f"{os.getcwd()}\\Upload\\{file}"

        # Get the text chunks of the PDF file, accumulate to the text_chunks list variable becaus load_and_split() returns a list of Document
        text_chunks += loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(
            chunk_size = 512,
            chunk_overlap = 30,
            length_function = len,
            separators= ["\n\n", "\n", ".", " "]
        ))

    return text_chunks

def text_chunks_hash_func(list_of_documents: list):
    list_of_documents = [doc.page_content for doc in list_of_documents]
    return tuple(list_of_documents)

# Function to create the vector store
@st.cache_data(hash_funcs={list: text_chunks_hash_func})
def get_vectorstore(text_chunks):
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    # embeddings = OpenAIEmbeddings(deployment="Xpose_pdf", model="text-embedding-ada-002", chunk_size=1)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=st.secrets['GOOGLE_API_KEY'])
    vectorstore = FAISS.from_documents(documents=text_chunks, embedding=embeddings)

    return vectorstore

def create_llm():
    # llm = AzureChatOpenAI(deployment_name="Test", # Name of the model deployment in Azure OpenAI
    #                       openai_api_version = os.getenv("OPENAI_API_VERSION"), 
    #                       openai_api_key = os.getenv("OPENAI_API_KEY"), 
    #                       openai_api_base = os.getenv("OPENAI_API_BASE"),
    #                       openai_api_type = os.getenv("OPENAI_API_TYPE"),
    #                       temperature=0.5
    #                       )
    
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv("GOOGLE_API_KEY"), 
                                 temperature=0.5, convert_system_message_to_human=True
                                )
    return llm

# Function to get the conversation chain
# Source: https://python.langchain.com/docs/modules/chains/popular/chat_vector_db#using-a-different-model-for-condensing-the-question
def get_conversation_chain(vectorstore):

    llm = create_llm()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, # Use the llm to generate the response, we can use better llm such as GPT-4 model from OpenAI to guarantee the quality of the response. For exp, the resopnse is more human-like
        retriever=vectorstore.as_retriever(),
        memory=memory,
        condense_question_llm=llm # Can use cheaper and faster model for the simpler task like condensing the current question and the chat history into a standalone question with GPT-3.5 if you are on budget. Otherwise, use the same model as the llm
    )
    return conversation_chain

# Function to remove files in the Upload folder
def remove_files():
    path = ".\\Upload"
    for file_name in os.listdir(path):
        # construct full file path
        file = path + '\\' + file_name
        if os.path.isfile(file) and file.endswith(".pdf"): # Only remove the PDF files
            print('Deleting file:', file)
            os.remove(file)

def reset_session_state():
    # Delete all the keys in session state
    for key in st.session_state.keys():
        if key != "file_uploader_key":
            del st.session_state[key]
    
    st.session_state["file_uploader_key"] += 1
    
    # Initialize the default session state variables again
    initialize_session_state()

# Function to initialize session states
def initialize_session_state():
    if "no_file_conversation" not in st.session_state:
        st.session_state.no_file_conversation = [
            SystemMessage(content="I'm a PDF Chatbot. Ask me a question about your documents!")
        ]
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        # st.write(bot_template.replace("{{MSG}}", "I'm a PDF Chatbot. Ask me a question about your documents!"), unsafe_allow_html=True)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if 'user_input_prompt' not in st.session_state: # This one is specifically use for clearing the user text input after they hit enter
        st.session_state.user_input_prompt = 'None'
    if "pdf_docs" not in st.session_state:
        st.session_state.pdf_docs = None
    if "file_uploader_key" not in st.session_state:
        st.session_state["file_uploader_key"] = 0
    if "is_processed" not in st.session_state:
        st.session_state.is_processed = None
    if "is_vectorstore" not in st.session_state:
        st.session_state.is_vectorstore = False
    if "extra_class" not in st.session_state: # This is a control variable, use to check the type of user's last conversation, either 'Warning' or 'None'
        st.session_state.extra_class = None


# Set the tab's title, icon and CSS style
page_icon = ":speech_balloon:"  # https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="PDF Chat App", page_icon=page_icon, layout="centered")
st.write(css, unsafe_allow_html=True)

# Load the environment variables
load_dotenv()

# Page header
st.header(body=f"Chat {page_icon} with your PDFs :books:")

# Main function
def main():

    # Layout of input/response containers
    response_container = st.container()
    colored_header(label='', description='', color_name='blue-30')
    input_container = st.container()

    # Initialize the session state variables
    initialize_session_state()

    # Print System Message at the brginning
    # if st.session_state.pdf_docs == None and st.session_state.is_processed == None:
    bot_welc_msg = st.session_state.get('no_file_conversation', [])[0]
    with response_container:
        st.write(bot_template.replace("{{MSG}}", bot_welc_msg.content), unsafe_allow_html=True)

    # ------------------------------------------------------ SIDEBAR ------------------------------------------------------ #
    # Sidebar contents
    with st.sidebar:

        # Upload PDF Files
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(label="Upload your PDFs here and click on 'Process'", accept_multiple_files=True, type=["pdf"], key=st.session_state["file_uploader_key"])
        process_button = st.button(label="Process")

        if process_button:
            if pdf_docs != []:
                # st.session_state.clear() # Clear the session state variables
                with st.spinner(text="Processing PDFs..."):
                    
                    # Save the uploaded files (PDFs) to the "Upload" folder
                    for pdf in pdf_docs:
                        save_uploadedfile(pdf)

                    # Get the text chunks from the PDFs
                    text_chunks = get_pdf_text_chunks()

                    # Create Vector Store
                    vectorstore = get_vectorstore(text_chunks)

                    # Create conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.session_state.is_vectorstore = True

                    # Remove the PDFs from the Upload folder
                    remove_files()
                
                # Print System Message at the end
                st.success(body=f"PDFs uploaded successfully!", icon="✅")

            # Use to check if PDFs are processed. If not uploaded and processed, users will be asked to upload PDFs when ask questions.
            st.session_state.is_processed = process_button

        if pdf_docs != []:
            # Use to check if PDFs are uploaded
            st.session_state.pdf_docs = pdf_docs
        else:
            st.session_state.pdf_docs = None
            st.session_state.is_vectorstore = False

        add_vertical_space(num_lines=1)

        # Web App References
        st.markdown('''
        ### About
        This app is an LLM-powered chatbot built using:
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
        - [OpenAI gpt-3.5-turbo](https://platform.openai.com/docs/models/gpt-3-5)
        - [Azure OpenAI Tutorial](https://techcommunity.microsoft.com/t5/startups-at-microsoft/build-a-chatbot-to-query-your-documentation-using-langchain-and/ba-p/3833134)
        - [YouTube Tutorial](https://youtu.be/dXxQ0LR-3Hg)
        ''')
        st.write("Made ❤️ by Lim Ming Jun")

        # Reset button part
        reset = st.sidebar.button('Reset all', on_click=reset_session_state)
        if reset:
            # for key in st.session_state.keys():
            #     if key != "file_uploader_key":
            #         del st.session_state[key]
            # st.session_state["file_uploader_key"] += 1
            # initialize_session_state()
            st.rerun()

# ------------------------------------------------------ MAIN PAGE ------------------------------------------------------ #
    # # Create the alert (Soon, this feature will be available for use)
    # error = st.alert("Please **upload your PDFs** before querying!", title="Error", type="error", icon="🚨")

    # User input
    with input_container:
        # User's text input was stored in "st.session_state.user_input_prompt"
        st.text_input(label="Ask a question about your documents", key="user_input", on_change=submit)
    
    if len(st.session_state.user_input_prompt) > 0: # If the user has entered something and not whitespace only
        # Check if the user has not uploaded the processed PDFs
        if st.session_state.user_input_prompt != 'None' and st.session_state.user_input_prompt.isspace() == False:
            if st.session_state.pdf_docs != None and st.session_state.is_processed != None and st.session_state.is_vectorstore == True:
                st.session_state.extra_class = 'bot-no-warning'
                with st.spinner('Generating...'):
                    with response_container:
                        handle_user_input(st.session_state.user_input_prompt) # st.session_state.user_input_prompt is "user input question"
            else:
                # Check if the user has just dropped the processed PDFs, and if so, remove the previous warning messages if there are any
                if st.session_state.extra_class == 'bot-no-warning':
                    st.session_state.no_file_conversation = [st.session_state.no_file_conversation[0]]
                
                warning_message = "Please upload your PDFs and click the Process button before asking a question!"
                st.session_state.extra_class = "bot-upload-warning"
                with st.spinner('Generating...'):
                    with response_container:
                        handle_user_input(st.session_state.user_input_prompt, warning_message, st.session_state.extra_class)
                # error.open()
        else:
            if st.session_state.user_input_prompt.isspace():
                st.session_state.user_input_prompt = 'None'
            with st.spinner('Generating...'):
                with response_container:
                    if st.session_state.extra_class == 'bot-no-warning':
                        handle_user_input(st.session_state.user_input_prompt)
                    elif st.session_state.extra_class == 'bot-upload-warning':
                        handle_user_input(st.session_state.user_input_prompt, "Please upload your PDFs before asking a question!", st.session_state.extra_class)
                    else:
                        pass
    
        # Clear the user input after the user hits enter
        st.session_state.user_input_prompt = 'None'


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    if st.runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
