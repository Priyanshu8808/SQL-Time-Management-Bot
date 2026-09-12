from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_agent
from langchain_community.document_loaders import PyPDFLoader,PyPDFDirectoryLoader
from langchain_community.vectorstores import InMemoryVectorStore
import streamlit as st
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver




## data in st session

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded= False

if "agent" not in st.session_state:
    st.session_state.agent= None

if "vector_store" not in st.session_state:
    st.session_state.vector_store= None 

if "messages" not in st.session_state:
    st.session_state.messages= []           



def process_doc(path):

    #load the documents
    loader=PyPDFDirectoryLoader(path)
    docs=loader.load()


    #split into chunks
    splitter=RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    docs=splitter.split_documents(documents=docs)


    #embedding and vecotr db
    emb=OllamaEmbeddings(model="nomic-embed-text")
    vector_db=InMemoryVectorStore.from_documents(
        documents=docs,
        embedding=emb
    )


    #llm prompt agents

    llm=ChatGroq(model="openai/gpt-oss-20b")

    #tool
    @tool
    def retrieve_context(query:str):
        """Retrieve documents relevant to a query from the knowledge base."""
        
        context=""
        docs=vector_db.similarity_search(query=query,k=3)
        for doc in docs:
            context+=doc.page_content + "\n"
        return context  

    #prompt

    system_prompt="""You are a helpful assistant that answers questions using retrieved context. 
            My knowledge base consists of the details from the uploaded document. 
            ALWAYS use the `retrieve_context` tool for questions requiring external knowledge.""" 

    memory=InMemorySaver()

    #CREATE AGENT
    agent=create_agent(
        model=llm,
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=memory
    )
    st.session_state.agent=agent
    st.session_state.document_uploaded=True  


#ui section 
#upload ui:
st.title("PDF - Assistant")


if not st.session_state.document_uploaded:
    uploaded=st.file_uploader(label="Select pdf file", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        with st.spinner("procesing..."):
            path="./doc_files/"
            for file in uploaded:
                with open(path + file.name, "wb" )as f:
                    f.write(file.getvalue())
                
            process_doc(path)
            st.rerun()        
            


#chat ui:
if st.session_state.document_uploaded and st.session_state.agent:
    for message in st.session_state.messages:
        role = message.get("role")
        content = message.get("content")
        st.chat_message(role).markdown(content)


    query = st.chat_input("Ask anything related to uploaded documents....")
    if query:
        st.session_state.messages.append({"role":"user", "content":query})

        st.chat_message("user").markdown(query)
        response = st.session_state.agent.invoke(
            {"messages":[{"role":"user", "content":query}]},
            {"configurable":{"thread_id":1}}
        )

        answer = response["messages"][-1].content
        st.chat_message("ai").markdown(answer)
        st.session_state.messages.append({"role":"ai", "content":answer})



    

    