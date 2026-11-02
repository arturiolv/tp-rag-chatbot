"""
TP : Chatbot basé sur un RAG (Retrieval-Augmented Generation)
"""

import os
import time
from functools import lru_cache

import requests
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# --- Configuration (fournie) -----------------------------------------------------
# Fournisseur : "openai", "mistral" ou "local" (question 36, sans clé API).
# Par défaut, détecté selon la clé présente dans .env.
PROVIDER = os.getenv("LLM_PROVIDER") or ("openai" if os.getenv("OPENAI_API_KEY") else "mistral")
PROVIDER = PROVIDER.strip().lower()
DEFAULT_MODELS = {
    "openai": ("gpt-6-luna", "text-embedding-3-small"),
    "mistral": ("mistral-small-latest", "mistral-embed"),
    "local": ("Qwen/Qwen2.5-1.5B-Instruct", "intfloat/multilingual-e5-small"),
}
if PROVIDER not in DEFAULT_MODELS:
    raise ValueError(f"LLM_PROVIDER inconnu : '{PROVIDER}' (attendu : openai, mistral ou local)")
CHAT_MODEL = os.getenv("CHAT_MODEL", DEFAULT_MODELS[PROVIDER][0])
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", DEFAULT_MODELS[PROVIDER][1])
# Une base par fournisseur : des embeddings de modèles différents ne sont pas compatibles
PERSIST_DIR = os.getenv("CHROMA_DIR", f"chroma_db_{PROVIDER}")

PAGES = [
    "Intelligence_artificielle_générative",
    "Transformeur_génératif_préentraîné",
    "Google_Gemini",
    "Grand_modèle_de_langage",
    "ChatGPT",
    "LLaMA",
    "Réseaux_antagonistes_génératifs",
    "Apprentissage_auto-supervisé",
    "Apprentissage_par_renforcement",
    "DALL-E",
    "Midjourney",
    "Stable_Diffusion"
]


# --- Modèles (fournis) : utilisez get_embeddings() et get_llm() dans votre code ---

def get_embeddings():
    """Renvoie le modèle d'embeddings du fournisseur choisi."""
    if PROVIDER == "local":
        from langchain_huggingface import HuggingFaceEmbeddings
        # E5 attend un préfixe différent pour les questions et pour les passages
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL,
                                     encode_kwargs={"normalize_embeddings": True, "prompt": "passage: "},
                                     query_encode_kwargs={"normalize_embeddings": True, "prompt": "query: "})
    if PROVIDER == "mistral":
        from langchain_mistralai import MistralAIEmbeddings
        from tokenizers import Regex, Tokenizer, models, pre_tokenizers
        # Sans tokenizer fourni, langchain-mistralai tente d'en télécharger un sur Hugging Face
        # (dépôt protégé, échoue derrière un proxy). Il ne sert qu'à grouper les textes par lots :
        # on utilise un tokenizer local « 1 caractère = 1 token », qui surestime volontairement.
        tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
        # max_retries / wait_time : l'offre gratuite limite le débit, on réessaie en cas d'erreur 429
        return MistralAIEmbeddings(model=EMBEDDING_MODEL, tokenizer=tokenizer,
                                   max_retries=10, wait_time=5)
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def get_llm():
    """Renvoie le modèle de chat du fournisseur choisi."""
    if PROVIDER == "local":
        return get_local_llm()
    if PROVIDER == "mistral":
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(model=CHAT_MODEL, temperature=0, max_retries=5)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=CHAT_MODEL)


_local_chat = {}


def get_local_llm(max_new_tokens=300):
    """Question 36 : modèle de chat exécuté localement, utilisable dans une chaîne LCEL, avec streaming."""
    from threading import Thread

    import torch
    import transformers
    from langchain_core.runnables import RunnableLambda
    from packaging import version
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

    if not _local_chat:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        dtype_kw = "dtype" if version.parse(transformers.__version__) >= version.parse("4.56") else "torch_dtype"
        _local_chat["tokenizer"] = AutoTokenizer.from_pretrained(CHAT_MODEL)
        _local_chat["model"] = AutoModelForCausalLM.from_pretrained(CHAT_MODEL, **{dtype_kw: dtype}).to(device).eval()
    tokenizer, model = _local_chat["tokenizer"], _local_chat["model"]
    roles = {"system": "system", "human": "user", "ai": "assistant"}

    def generate(prompt_value):
        conversation = [{"role": roles[m.type], "content": m.content} for m in prompt_value.to_messages()]
        text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        batch = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(**batch, streamer=streamer, max_new_tokens=max_new_tokens, do_sample=False,
                      temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
                      pad_token_id=tokenizer.eos_token_id)
        Thread(target=model.generate, kwargs=kwargs).start()
        yield from streamer

    return RunnableLambda(generate)


def get_wikipedia_page(title: str):
    """
    Retrieve the full text content of a Wikipedia page.

    :param title: str - Title of the Wikipedia page.
    :return: str - Full text content of the page as raw string.
    """
    URL = "https://fr.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "redirects": 1,  # suit les redirections (sinon une page redirigée revient vide)
    }
    headers = {"User-Agent": "RAG_project/0.0.1"}

    response = requests.get(URL, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    data = response.json()

    page = next(iter(data["query"]["pages"].values()))
    return page["extract"] if "extract" in page else None


@lru_cache(maxsize=None)
def fetch_page(title, retries=5, wait=10):
    """Wikipedia limite le débit (erreur 429) : on attend puis on réessaie, et on garde les pages déjà obtenues."""
    for attempt in range(retries):
        try:
            return get_wikipedia_page(title)
        except requests.HTTPError as error:
            if error.response.status_code != 429 or attempt == retries - 1:
                raise
            time.sleep(wait * (attempt + 1))


def load_documents():
    """
    Partie 1 — Télécharge toutes les pages de PAGES et renvoie une liste de Document
    (texte + métadonnées "title" et "source").
    """
    documents = []
    for page in PAGES:
        try:
            text = fetch_page(page)
        except requests.RequestException as error:
            print(f"Page ignorée (erreur réseau) : {page} ({error})")
            continue
        if not text:
            print(f"Page ignorée (vide) : {page}")
            continue
        documents.append(Document(page_content=text,
                                  metadata={"title": page.replace("_", " "),
                                            "source": f"https://fr.wikipedia.org/wiki/{page}"}))
    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Partie 2 — Découpe les documents en chunks qui se chevauchent ; les métadonnées sont conservées."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


def get_vectorstore(chunk_size=1000, chunk_overlap=200, persist_dir=None):
    """Partie 3 — Ouvre la base Chroma ; ne calcule les embeddings que si elle est vide."""
    vectorstore = Chroma(collection_name="wikipedia_ia_generative",
                         embedding_function=get_embeddings(),
                         persist_directory=persist_dir or PERSIST_DIR)
    if vectorstore._collection.count() == 0:
        vectorstore.add_documents(split_documents(load_documents(), chunk_size, chunk_overlap))
    return vectorstore


def get_retriever(k=4, **kwargs):
    """
    Parties 2 et 3 — Découpe les documents, calcule les embeddings, les stocke dans
    Chroma (persisté dans PERSIST_DIR) et renvoie un retriever.
    """
    return get_vectorstore(**kwargs).as_retriever(search_kwargs={"k": k})


def format_docs(docs):
    """Partie 4 — Concatène les chunks récupérés en un seul texte de contexte."""
    return "\n\n---\n\n".join(f"[{i}] Page « {doc.metadata.get('title', 'inconnue')} »\n{doc.page_content}"
                              for i, doc in enumerate(docs, 1))


PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Tu es un assistant qui répond aux questions sur l'IA générative. Réponds en français, uniquement à partir "
     "du contexte ci-dessous. Si la réponse n'y figure pas, dis que tu ne sais pas.\n\nContexte :\n{context}"),
    ("human", "{question}"),
])

_default_retriever = None


def rag_answer(query, retriever=None, stream=True, prompt=PROMPT):
    """
    Partie 4 — Récupère les chunks pertinents, interroge le LLM et affiche
    la réponse puis ses sources. Renvoie la réponse et les documents utilisés.
    """
    global _default_retriever
    if retriever is None:
        _default_retriever = _default_retriever or get_retriever()
        retriever = _default_retriever
    docs = retriever.invoke(query)
    chain = prompt | get_llm() | StrOutputParser()
    inputs = {"context": format_docs(docs), "question": query}

    if stream:
        tokens = []
        for token in chain.stream(inputs):
            print(token, end="", flush=True)
            tokens.append(token)
        answer = "".join(tokens)
        print("\n\nSources :")
        for source in dict.fromkeys(doc.metadata["source"] for doc in docs):
            print(f"- {source}")
    else:
        answer = chain.invoke(inputs)
    return answer, docs


if __name__ == "__main__":
    print(f'Fournisseur : {PROVIDER} (chat : {CHAT_MODEL}, embeddings : {EMBEDDING_MODEL})')
    print('Building retriever...')
    retriever = get_retriever()

    try:
        while True:
            print('-' * 50)
            print('Posez une question :')
            question = input('> ')
            print()
            rag_answer(question, retriever)
            print('\n')

    except (KeyboardInterrupt, EOFError):
        print("\nExiting...")
