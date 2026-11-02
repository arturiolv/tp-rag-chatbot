"""Interface web du chatbot RAG : streamlit run app.py"""

import os

import streamlit as st

# En ligne, il n'y a pas de fichier .env : la clé vient de st.secrets. Il faut la copier avant d'importer main.
for key in ("MISTRAL_API_KEY", "OPENAI_API_KEY"):
    try:
        if key in st.secrets:
            os.environ[key] = st.secrets[key]
    except Exception:  # en local, sans .streamlit/secrets.toml, l'accès à st.secrets lève une exception
        pass

from main import get_retriever, rag_answer  # noqa: E402


@st.cache_resource(show_spinner="Préparation de la base documentaire...")
def load_retriever():
    # Exécuté une seule fois, et non à chaque interaction
    return get_retriever()


def show_answer(message):
    st.markdown(message["content"])
    sources = message.get("sources", [])
    if sources:
        st.markdown("**Sources :** " + " · ".join(f"[{title}]({url})" for title, url in sources))
    if message.get("passages"):
        with st.expander("Passages récupérés"):
            for title, text in message["passages"]:
                st.markdown(f"**{title}**")
                st.text(text)


st.title("Chatbot RAG sur l'IA générative")
st.caption("Réponses fondées sur 12 pages Wikipedia en français, avec leurs sources.")

retriever = load_retriever()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        show_answer(message)

question = st.chat_input("Posez une question")
if question:
    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        show_answer(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Recherche et rédaction de la réponse..."):
            answer, docs = rag_answer(question, retriever, stream=False)
        sources = dict.fromkeys((doc.metadata["title"], doc.metadata["source"]) for doc in docs)
        assistant_message = {"role": "assistant", "content": answer, "sources": list(sources),
                             "passages": [(doc.metadata["title"], doc.page_content) for doc in docs]}
        show_answer(assistant_message)
    st.session_state.messages.append(assistant_message)
