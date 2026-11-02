# TP : Chatbot basé sur un RAG

Chatbot qui répond à des questions sur l'IA générative à partir de 12 pages Wikipedia en français (LangChain, Chroma, Streamlit).

## Installation

```
python -m venv .venv
source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install -r requirements.txt
cp .env.template .env          # puis renseigner OPENAI_API_KEY ou MISTRAL_API_KEY
```

Ne partagez jamais le fichier `.env` : il contient une clé API, et il est exclu de Git par `.gitignore`.

## Lancement

Dans le terminal :

```
python main.py
```

Avec l'interface web :

```
streamlit run app.py
```

La première exécution télécharge les pages et calcule les embeddings. Les lancements suivants réutilisent la base stockée dans `chroma_db_<fournisseur>/`.

## Fournisseurs

| `LLM_PROVIDER` | Modèle de chat | Embeddings | Clé API |
|---|---|---|---|
| `openai` | `gpt-6-luna` | `text-embedding-3-small` | `OPENAI_API_KEY` |
| `mistral` | `mistral-small-latest` | `mistral-embed` | `MISTRAL_API_KEY` |
| `local` | `Qwen/Qwen2.5-1.5B-Instruct` | `intfloat/multilingual-e5-small` | aucune, GPU conseillé |

Le fournisseur `local` demande aussi : `pip install langchain-huggingface sentence-transformers transformers torch`.

## Déploiement sur Streamlit Community Cloud

Indiquez ce dépôt, la branche `main` et le fichier `app.py`, puis renseignez la clé dans les secrets de l'application, au format TOML :

```
OPENAI_API_KEY = "votre-cle"
```

L'application est publique : toute personne qui a le lien consomme les crédits de la clé.

## Docker

Le `Dockerfile` n'a pas été testé.

```
docker build -t tp-rag .
docker run -p 8501:8501 -e OPENAI_API_KEY=... tp-rag
```
