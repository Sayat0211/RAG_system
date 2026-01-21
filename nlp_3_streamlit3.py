import os
import tempfile
import streamlit as st
import PyPDF2
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from typing import List, Dict
import hashlib
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")
@st.cache_resource
def init_llm_model():
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-flash')
class SimpleRAG:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory=persist_directory
        self.embedding_model=load_embedding_model()
        self.llm=init_llm_model()
        self.client=chromadb.PersistentClient(path=persist_directory)
        try:
            self.collection=self.client.get_collection(name="main_documents")
            st.success("vector database loaded")
        except:
            self.collection = self.client.create_collection(name="main_documents")
            st.info("New vector database created")
        self.chunk_size=400
        self.chunk_overlap=50
    @st.cache_data(show_spinner=False)
    def _process_pdf_content(_self, pdf_path, filename):
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text+=f"Page {page_num + 1}: {page_text}\n"
                return text, len(pdf_reader.pages)
        except Exception as e:
            st.error(f"PDF download error: {e}")
            return "", 0
    def _get_file_hash(self,file_path):
        hasher=hashlib.md5()
        with open(file_path,'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()
    def _is_file_processed(self, file_hash):
        existing_metadatas = self.collection.get()['metadatas']
        for metadata in existing_metadatas:
            if metadata.get('file_hash') == file_hash:
                return True
        return False
    def load_and_process_pdf(self, pdf_path):
        return self._process_pdf_content(pdf_path, "pdf_processing")
    def chunk_text(self, text):
        words=text.split()
        chunks=[]
        for i in range(0, len(words),self.chunk_size-self.chunk_overlap):
            chunk=" ".join(words[i:i+self.chunk_size])
            chunks.append(chunk)
            if i+self.chunk_size>=len(words):
                break
        return chunks
    def generate_embeddings(self,chunks):
        return self.embedding_model.encode(chunks)
    def add_document_to_store(self,pdf_path,filename):
        file_hash=self._get_file_hash(pdf_path)
        if self._is_file_processed(file_hash):
            st.warning(f"File '{filename}' have been processed")
            return False
        text,page_count=self.load_and_process_pdf(pdf_path)
        if not text:
            return False
        chunks=self.chunk_text(text)
        if not chunks:
            st.error("Couldn't split text into chunks")
            return False
        embeddings = self.generate_embeddings(chunks)
        start_id = self.collection.count()
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            self.collection.add(
                embeddings=[embedding.tolist()],
                documents=[chunk],
                metadatas=[{
                    "chunk_id":start_id+i,
                    "source":filename,
                    "file_hash":file_hash,
                    "page_count":page_count
                }],
                ids=[f"chunk_{start_id+i}"]
            )       
        st.success(f"File '{filename}' successfully added! ({page_count} pages, {len(chunks)} chunks)")
        return True 

    def search_documents(self, query, top_k=3):
        query_embedding=self.embedding_model.encode([query]).tolist()
        results=self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        return results
    def generate_answer(self,query,context_chunks,sources):
        context = "\n\n".join([f"[Fragment {i+1}] {chunk}" for i, chunk in enumerate(context_chunks)])
        source_info = "\n".join([f"-{source}" for source in set(sources)])
        prompt = f"""You are an assistant that answers questions based on the provided context.
        CONTEXT FOR THE ANSWER:
        {context}
        SOURCE FILES:
        {source_info}
        USER'S QUESTION: {query}
        Requirements:
        1) Answer ONLY based on the information from the context.Users questions may be rehashed
        2) If the information for the answer is not in the context, honestly say: "Information not found in the documents."
        3) Be precise and use facts from the context.
        4) The answer must be in the same language as the user's question.
        5) Mention which source files were used for the answer.
        ANSWER:"""
        try:
            response = self.llm.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating answer:{e}"

    def ask_question(self,query,top_k=3):
        search_results=self.search_documents(query,top_k)
        context_chunks=search_results['documents'][0]
        metadatas=search_results['metadatas'][0]
        if context_chunks:
            st.subheader("Relevant fragments found:")
            for i, (chunk, metadata) in enumerate(zip(context_chunks, metadatas)):
                st.write(f"**Fragment {i+1}** ({metadata.get('source','Unknown')}):")
                st.text_area(f"Contents of the fragment {i+1}:", 
                           chunk, height=150, key=f"chunk_{i}")
                st.write("---")
        if not context_chunks:
            return "No relevant information was found in the documents.",[]
        sources=[metadata.get('source','Unknown') for metadata in metadatas]
        answer=self.generate_answer(query,context_chunks,sources)
        return answer,sources
    def get_stats(self):
        count=self.collection.count()
        metadatas=self.collection.get()['metadatas']
        unique_files=set(metadata.get('source','Unknown') for metadata in metadatas)
        return {
            "total_chunks":count,
            "unique_files":len(unique_files),
            "file_list":list(unique_files)
        }
def main():
    st.set_page_config(
        page_title="RAG System",
        layout="wide"
    )
    st.title("Search System")
    st.markdown("Upload files and ask questions based on them")
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system=SimpleRAG()
    with st.sidebar:
        st.header("Document management")
        uploaded_files=st.file_uploader(
            "Download PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Select PDF files to add to the system"
        )
        if uploaded_files:
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path=tmp_file.name
                if st.session_state.rag_system.add_document_to_store(tmp_path, uploaded_file.name):
                    os.unlink(tmp_path)
        st.header("Statistics")
        stats = st.session_state.rag_system.get_stats()
        st.metric("Amount of Chunks",stats["total_chunks"])
        st.metric("Quantity of files",stats["unique_files"])
        if stats["file_list"]:
            st.subheader("Uploaded files:")
            for file in stats["file_list"]:
                st.write(f"• {file}")
        st.header("SETTINGS")
        if st.button("Clear the entire database", type="secondary"):
            try:
                st.session_state.rag_system.client.delete_collection("main_documents")
                st.session_state.rag_system.collection=st.session_state.rag_system.client.create_collection(name="main_documents")
                st.success("Data base deleted!")
                st.rerun()
            except Exception as e:
                st.error(f"Error while cleaning: {e}")
    col1, col2 = st.columns([2, 1])    
    with col2:
        st.header("INFORMATION")
        st.info("""
        **How to use:**
        1. Upload PDF files via the sidebar
        2. Enter your question in the text field
        3. Click “Ask a question”
        """)
    with col1:
        st.header("Ask a question:")
        question = st.text_area(
            "Enter your question:",
            placeholder="For example: What are the criteria for receiving a Michelin star?",
            height=100
        )
        col_a, col_b = st.columns(2)
        with col_a:
            top_k = st.slider("Number of relevant fragments",1,10,3)
        with col_b:
            if st.button("Ask a question",type="primary",use_container_width=True):
                if question.strip():
                    with st.spinner("loading..."):
                        st.session_state.last_question = question
                        answer, sources = st.session_state.rag_system.ask_question(question, top_k)
                        st.subheader("ANSWER:")
                        st.write(answer)
                        if sources:
                            st.subheader("Sources used:")
                            for source in set(sources):
                                st.write(f"• {source}")
                else:
                    st.warning("Please enter your question")  
        st.subheader("Quick questions")
        quick_questions = [
            "What are the criteria for receiving a Michelin star?",
            "Who are the Michelin inspectors?",
            "What is Bib Gourmand?",
            "When was the first Michelin Guide created?"
        ]
        for q in quick_questions:
            if st.button(q, use_container_width=True):
                st.session_state.quick_question = q
                st.rerun()
    if 'quick_question' in st.session_state:
        question = st.session_state.quick_question
        del st.session_state.quick_question
        with st.spinner("loading..."):
            st.session_state.last_question = question
            answer, sources = st.session_state.rag_system.ask_question(question, top_k)
            st.subheader("ANSWER:")
            st.write(answer)
            if sources:
                st.subheader("Sources used:")
                for source in set(sources):
                    st.write(f"•{source}")

if __name__ == "__main__":
    main()

