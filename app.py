import streamlit as st
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain import hub
from gradio.themes.base import Base
import gradio as gr
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

embeddings = OpenAIEmbeddings()
db3 = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = db3.as_retriever()

prompt = hub.pull("coty/rag-promptv1")
llm = ChatOpenAI(model_name="gpt-4-1106-preview", max_tokens=1024)

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # 确保这里的chain_type是有效的
    retriever=compression_retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True)

def get_answer(question):
    data = qa.invoke(question)
    return data



# Streamlit 应用界面
st.title("LangChain QA System")
question = st.text_input("请输入您的问题：", "")
if st.button("获取答案"):
    if question:
        data = get_answer(question)
        st.write("问题：", data['query'])
        st.write("答案：", data['result'])
        
        # 检查是否有检索到的文档
        if data['source_documents']:
            st.write("-" * 50)  # 分隔每个文档
            st.write("来源：", data['source_documents'][0].metadata['source'])
            st.write(data['source_documents'][0].page_content)
        else:
            st.write("未检索到相关信息。")
    else:
        st.write("请输入一个问题。")