import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()

# Initialize services
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Pinecone index configuration
index_name = "physical-therapy"
index = pc.Index(index_name)

# Initialize or load message history
if 'message_history' not in st.session_state:
    st.session_state.message_history = []

def generate_openai_response(prompt, temperature=0.7):
    """Generates a response from OpenAI based on a structured prompt."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant designed to support physical therapists..."},
                {"role": "user", "content": prompt}
            ] + [
                {"role": "user" if msg['sender'] == 'You' else "assistant", "content": msg['content']}
                for msg in st.session_state.message_history
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {str(e)}"

def search_similar_documents(query, top_k=5):
    """Searches for documents in Pinecone that are similar to the query."""
    query_vector = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    vector = query_vector.data[0].embedding
    results = index.query(vector=vector, top_k=top_k, include_metadata=True)
    contexts = [x['metadata']['text'] for x in results['matches']]
    return contexts

def generate_prompt(query):
    """Generates a comprehensive prompt including contexts from similar documents."""
    prompt_start = "Answer the question based on the context below.\n\nContext:\n"
    prompt_end = f"\n\nQuestion: {query}\nAnswer:"
    similar_docs = search_similar_documents(query)
    
    # Compile contexts into a single prompt, respecting character limits
    prompt = prompt_start
    for doc in similar_docs:
        if len(prompt + doc + prompt_end) < 3750:
            prompt += "\n\n---\n\n" + doc
        else:
            break
    prompt += prompt_end
    return prompt

st.title("EclatAI")

user_input = st.text_input("You: ", "")

if user_input:
    # Add user's message to history
    st.session_state.message_history.append({"sender": "You", "content": user_input})

    final_prompt = generate_prompt(user_input)
    bot_response = generate_openai_response(final_prompt)
    
    # Add Aidin's response to history
    st.session_state.message_history.append({"sender": "Doc", "content": bot_response})

    # Display chat messages from history on app rerun
    for message in st.session_state.message_history:
        role = "user" if message["sender"] == "You" else "assistant"
        with st.chat_message(role):
            st.markdown(message["content"])