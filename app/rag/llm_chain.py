from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_community.vectorstores import FAISS
from app.core.config import settings

def create_rag_chain(vector_store: FAISS):
    """
    Creates the full RAG chain for conversation and question-answering.
    """
    llm = ChatGroq(model=settings.LLM_MODEL, api_key=settings.GROQ_API_KEY)

    # Contextualize question prompt: This reformulates the user's question to be a standalone question
    # based on chat history. This is useful for follow-up questions.
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(
        llm, vector_store.as_retriever(), contextualize_q_prompt
    )

    # Answering prompt: This prompt takes the retrieved documents (context) and the user's question
    # to generate a final, concise answer.
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, just say "
        "that you don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Chain that combines the documents into a single string to be passed to the LLM.
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # The final retrieval chain that connects the retriever and the question-answering chain.
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain
