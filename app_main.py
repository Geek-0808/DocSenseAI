import os,re
from pandas.core.indexes.base import JoinHow
import requests
import streamlit as st
import json
import urllib.parse
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from typing import List, Optional, Dict, Any
from langchain.text_splitter import CharacterTextSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.retrievers import BaseRetriever
from langchain.schema import Document as LC_Document
from htmlTemplates import css, bot_template, user_template
from youtube_transcript_api import YouTubeTranscriptApi
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
)
from llama_index.core import load_index_from_storage, StorageContext , Document
from llama_index.core.schema import TextNode
from pinecone import Pinecone, ServerlessSpec
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex
# from langchain_community.retrievers.llama_index import LlamaIndexRetriever
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# // ✨ Google Programmable Search Integration
# function searchCSE(query) {
#   const apiKey = "AIzaSyDMAPSzitB9Aq1vb6Y3hQgBDBTViMn2qMk"; // 🔒 Replace with your API key
#   const cx = "5427ac308d5524f0e";             // 🔒 Replace with your CSE ID
#
#   const url = `https://www.googleapis.com/customsearch/v1?q=${encodeURIComponent(query)}&key=${apiKey}&cx=${cx}`;
#   // const url = `https://cse.google.com/cse?cx=5427ac308d5524f0e`
#
#   try {
#     const response = UrlFetchApp.fetch(url);
#     const data = JSON.parse(response.getContentText());
#     // return data.items?.[0]?.snippet || "No info found.";
#     if (!data.items || data.items.length === 0) return "No search results.";
#
#     // Combine top 3 snippets for more context
#     const snippets = data.items.slice(0, 10).map(item => item.snippet).join(" | ");
#     return snippets;
#   } catch (error) {
#     return "Error fetching from CSE.";
#   }
# }
# Ensure the transcript directory exists
os.makedirs('transcript', exist_ok=True)

def process_youtube_url(youtube_url):
    video_id = extract_video_id(youtube_url)
    if not video_id:
        st.error("Video ID not found. Please provide a valid YouTube link.")
        return None

    transcript_text = get_video_transcript(video_id)
    if not transcript_text:
        st.error("Transcript can't be extracted. Please try another link.")
        return None

    output_file = f'transcript/{video_id}.txt'
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(transcript_text)
    
    return output_file

def extract_video_id(youtube_url):
    video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', youtube_url)
    if video_id_match:
        return video_id_match.group(1)
    return None

def get_video_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return "\n".join([entry['text'] for entry in transcript])
    except Exception as e:
        return None

def get_video_text():
    None

def get_search_results(query):
    query = urllib.parse.quote(query)
    apiKey = "AIzaSyCEl9eYcYJr7b-aXxYEWqZF2Mcr6Uq1Ogk" # 🔒 Replace with your API key
    cx = "26c45e1df0e6e457a"             # 🔒 Replace with your CSE ID
    url = f'https://www.googleapis.com/customsearch/v1?q={query}&key={apiKey}&cx={cx}'
    res = requests.get(url)
    # search = GoogleSearchResults({"q": query})
    # results = search.get_dict()
    res = json.loads(res.text)
    snippets = []
    for i in res['items']:
        snippets.append({'title':i['title'], 'text': i['snippet']})
    print(f'Query: {query} | Result: {snippets}')
    return snippets 

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# def get_text_chunks(text):
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#     )
#     chunks = text_splitter.split_text(text)
#     print(f"{'_'*40}\nChunks : {chunks}{'_'*40}\n")
#     return chunks

def get_text_chunks(text, embed_model, file_name):
    splitter = SemanticSplitterNodeParser(
        # buffer_size=1, 
        chunk_size=1000,
        # breakpoint_percentile_threshold=95, 
        embed_model=embed_model
        )
    doc = Document(text=text, id_=str(file_name))
    nodes = splitter.get_nodes_from_documents([doc])
    # newNodes = nodes
    newNodes = []
    for i,n in enumerate(nodes):
        new_node = TextNode(
            text=n.text,
            id_=f"{file_name}#{i}",
            # index_id=,
            metadata=doc.metadata
            # obj=doc
        )
        newNodes.append(new_node)
        # newNodes.append(n.text)
    # print(f"{'_'*40}\nChunks : {newNodes}{'_'*40}\n")
    return newNodes


def get_vectorstore(text_chunks, embeddings):
    # embeddings = OpenAIEmbeddings()
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

class LlamaIndexRetriever(BaseRetriever):
    def __init__(
        self,
        index: VectorStoreIndex,
        *,
        similarity_top_k: int = 10,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **retriever_kwargs: Any
    ):
        # 1️⃣ Initialize the BaseRetriever so self.tags, self.metadata exist
        super().__init__(tags=tags, metadata=metadata)

        # 2️⃣ Grab the LlamaIndex retriever under the covers
        self._inner = index.as_retriever(
            similarity_top_k=similarity_top_k,
            similarity_threshold=0.7, 
            **retriever_kwargs
        )

    def _get_relevant_documents(self, query: str) -> List[LC_Document]:
        # 3️⃣ Call the LlamaIndex retriever
        nodes = self._inner.retrieve(query)
        # print(f"{'_'*40}\nNodes: {nodes}\n{'_'*40}\n")
        # 4️⃣ Convert Nodes → LangChain Documents
        res =  [
            LC_Document(page_content=node.text, metadata={'title': node.id_.split('#')[0], 'score': node.score})
            for node in nodes if node.score >= 0.75
        ]
        return res 

    # (optional) if you also want async support:
    async def _aget_relevant_documents(
        self, query: str, **kwargs: Any
    ) -> List[LC_Document]:
        nodes = await self._inner.aretrieve(query, **kwargs)  # if available
        return [
            LC_Document(page_content=node.text, metadata=node.metadata or {})
            for node in nodes
        ]

class FallbackRetriever(BaseRetriever):
    # def __init__(self, docs: List[LC_Document]):
        # self.docs = docs
    docs: List[LC_Document]

    def get_relevant_documents(self, query: str) -> List[LC_Document]:
        # Fallback logic to retrieve documents
        return self.docs

def get_conversation_chain(retriever, style_prompts, llm, style="Formal"):
    # retriever = index.as_retriever()
    # if not isinstance(retriever, BaseRetriever):
    #     print("Retriever is not a BaseRetriever instance.")
    # else:
    #     print("Retriever is a BaseRetriever instance.")
    system_prompt = style_prompts.get(style, "You are a helpful assistant . Answer the users questions based on the context provided")
    prompt = ChatPromptTemplate.from_messages([
        ('system', system_prompt),
        ('human', "Based on Context:\n{context}\n Answer:\n{question}"),
    ])
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    print(f'Type: {type(retriever)}| Ret: {retriever=}, LLM: {llm=},MEM: {memory=}')
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt" : prompt},
        # return_source_documents=True,
    )
    return conversation_chain

def handle_userinput(user_question, index, style_prompts, style='Formal', search=False):
    
    def get_search_retriever(user_question: str) -> BaseRetriever:
        search_results = get_search_results(user_question)
        docs = [LC_Document(page_content=search_result['text'], metadata={'title': search_result['title']}) for search_result in search_results]
        retriever = FallbackRetriever(docs=docs)
        return retriever 
  
    chain = st.session_state.conversation
    if style == 'Concise':
        st.session_state.max_tokens = 100
    else:
        st.session_state.max_tokens = 1000
    # new_llm = ChatOpenAI(model="gpt-3.5-turbo", max_tokens = st.session_state.max_tokens)
    # chain.question_generator.llm       = new_llm
    # chain.combine_docs_chain.llm       = new_llm
    # chain.combine_docs_chain.llm_chain.llm = new_llm
    chain.question_generator.llm.max_tokens = st.session_state.max_tokens
    chain.combine_docs_chain.llm_chain.llm.max_tokens = st.session_state.max_tokens
    print(f'Selected Style: {style}')
    print(f'Max Tokens: {st.session_state.max_tokens}')
    if search:
        st.write('Searching ....')
        chain.retriever = get_search_retriever(user_question)
    else:
        retriever = LlamaIndexRetriever(index=index)

        chunks = retriever.get_relevant_documents(user_question)
        print(f'Num: {len(chunks)} | Chunks: {chunks}')

        if len(chunks) == 0:
            print(f'NO CHUNKS FOUND, SEARCHING ....')
            chain.retriever = get_search_retriever(user_question)
        else:
            chain.retriever = retriever
    #   context = "\n\n".join(search_results)
    # else: 
    #     context = "\n\n".join([f"From {chunk.metadata} : {chunk.page_content}" for chunk in chunks])
    response = chain.invoke({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    li_embed_model = OpenAIEmbedding()
    lc_embed_model = OpenAIEmbeddings()
    st.session_state.max_tokens = 1000

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = "final-project"
    if index_name not in pc.list_indexes().names():
        pc.create_index(index_name, dimension=1536, metric="cosine",
                        spec=ServerlessSpec(
                            cloud="aws",
                            region="us-east-1"
                            )
                        )
    pinecone_index = pc.Index(index_name)
    print(pinecone_index.describe_index_stats())
    # pinecone_index.delete(delete_all=True)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
# construct vector store and customize storage context
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )
    PINECONE_ENVIRONMENT = "us-east-1-aws"
    # get_search_results('Cristiano Ronaldo')
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "index" not in st.session_state:
        st.session_state.index = None
    if "search" not in st.session_state:
        st.session_state.search = False

    st.header("Chat with multiple PDFs :books:")
    box, button = st.columns([4, 1])
    style_prompts = {
        "Formal": "You are a helpful assistant . Answer the users questions based on the context provided & strictly Respond in a clear, professional, and formal tone. Use precise language and avoid contractions. Be respectful and courteous. Use language that requires high-level vocabulary & advanced reading comprehension.",
        "Casual": "You are a helpful assistant . Answer the users questions based on the context provided & respond in a friendly and informal tone, like you're talking to a colleague or friend. Use plenty of emojis & slang language. Be relatable and approachable.",
        "Explanatory": "You are a helpful assistant . Answer the users questions based on the context provided & explain concepts in a detailed, step-by-step manner as if teaching a beginner. Expand upon each point & the idea, providing the user with a something akin to a complete & comprehensive report.",
        "Concise": "You are a helpful assistant . Answer the users questions based on the context provided & answer briefly and to the point. Include only essential details. Answer in a single sentence or two. Avoid unnecessary elaboration.",
    }
    st.subheader("Response Style")
    selected_style = st.selectbox(
        "Choose the tone of the response:",
        options=list(style_prompts.keys()),
        index=0  # default is 'Formal'
    )
    st.session_state.selected_style = selected_style
    with box:
        user_question = st.text_input("Ask a question about your documents:")
    with button:
        submit_button = st.button("Submit", type="primary")
    if user_question and submit_button:
        print(f'User Question: {user_question}')
        handle_userinput(user_question,st.session_state.index, style_prompts, style=selected_style, search=st.session_state.search)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
           "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
 
        video_url = st.text_input("Enter your Video URL")

        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = ""
                # get pdf text
                if pdf_docs:
                    raw_text += get_pdf_text(pdf_docs)  # assuming get_pdf_text takes a single file

                if video_url:
                    transcript_file = process_youtube_url(video_url)
                    if transcript_file:
                        with open(transcript_file, 'r', encoding='utf-8') as file:
                            raw_text += file.read()
                
                if not raw_text:
                    st.error("Please upload at least a PDF or provide a YouTube URL.")
                #else:
                    #st.success("Processing complete!")
                    #st.text_area("Combined Content", raw_text, height=400)

                # get the text chunks
                text_chunks = []
                for file in pdf_docs:
                    file_name = file.name
                    print(pdf_docs[0].name)
                    # text_chunks = get_text_chunks(raw_text, li_embed_model, file_name=file.name)
                    # text_chunks = get_text_chunks(raw_text, li_embed_model, file_name=file.name)
                    temp = get_text_chunks(raw_text, li_embed_model, file_name=file.name)
                    text_chunks.extend(temp)
                # create vector store
                # vectorstore = get_vectorstore(text_chunks, lc_embed_model)
                st.session_state.index = VectorStoreIndex(text_chunks, storage_context=storage_context)
                llm = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=st.session_state.max_tokens)
                ini_retriever = LlamaIndexRetriever(st.session_state.index)
                st.session_state.conversation = get_conversation_chain(ini_retriever, style_prompts, llm, style=selected_style)
        st.subheader("Mode")
        mode = st.radio(
            label="", 
            options=["Chat", "Search"], 
            index=0, 
            key="app_mode"
        )
        if mode == "Search":
            st.session_state.search = True
                # create conversation chain

if __name__ == '__main__':
    main()
