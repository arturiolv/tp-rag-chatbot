# Compte rendu : chatbot basé sur un RAG

**Dépôt GitHub :** https://github.com/arturiolv/tp-rag-chatbot

**Application déployée :** https://tp-rag-chatbot-arturiolv.streamlit.app/

**Modèles.** `gpt-6-luna` et `text-embedding-3-small`, sur 12 pages Wikipedia, soit 425 chunks.

**Question 26.** Après la modification du sous-titre et un `git push`, l'application en ligne a affiché le nouveau texte environ une minute plus tard, sans nouveau déploiement manuel.

## Questions de réflexion

**Question 1.** On ne donne pas les 12 pages entières au LLM pour trois raisons. Le coût : on paie chaque token, et la seule page ChatGPT fait 55 309 caractères. La limite et la lenteur : le corpus peut dépasser la fenêtre de contexte, et un long prompt ralentit la réponse. La qualité : l'information utile est noyée. Dans le Lab 3, un passage utile placé tard parmi 9 passages était moins bien lu (précision de 0,96 à 0,76).

**Question 8.** Le chevauchement évite de perdre une information coupée entre deux chunks. Des chunks trop petits perdent leur contexte, par exemple le nom du sujet : dans le Lab 3, à 100 caractères sans en-tête, le rappel tombe à 0,12. Des chunks trop grands mélangent plusieurs sujets : leur embedding est moins précis, et ils coûtent plus de tokens.

**Question 9.** La dimension mesurée est 1536 avec `text-embedding-3-small`, et 384 avec `multilingual-e5-small`. Selon la documentation, `mistral-embed` donne 1024 (non mesuré). On ne peut pas mélanger ces vecteurs : si les dimensions diffèrent, le calcul est impossible, et à dimension égale, chaque modèle a son propre espace, donc la proximité n'aurait aucun sens.

**Question 10.** Exemples raisonnés, non testés, car le projet n'a pas de recherche BM25. Les embeddings devraient mieux réussir sur « Quel outil fabrique des illustrations à partir d'une phrase ? » : peu de mots en commun avec « générer des images à partir de descriptions textuelles », mais un sens proche. BM25 devrait mieux réussir sur « Qu'est-ce que CreativeML OpenRAIL-M ? », un terme rare et exact.

**Question 17.** Avec la consigne, le chatbot répond « Je ne sais pas. » à la question sur la capitale de l'Australie. Sans la consigne, il répond « Canberra », alors que les pages récupérées n'en parlent pas : la réponse vient de son entraînement. Le retriever renvoie toujours k passages, même sans rapport, et c'est la consigne qui fait s'abstenir le modèle. Sans elle, une réponse fausse sortirait de la même façon, sans source pour la vérifier.

**Question 18.** Sur « Qui a développé Stable Diffusion ? », le chatbot répond souvent « le contexte ne précise pas », alors qu'on attend Stability AI. Deux des 4 passages récupérés citent Stability, mais pas comme développeur. Avec k = 8, il répond bien « Stability AI ». Le problème vient donc d'abord de **la recherche** : à k = 4, le passage utile n'est pas récupéré. La génération, elle, ne déduit pas plus que ce qui est écrit. Je n'ai pas relu un à un les passages de rang 5 à 8. Le modèle n'est pas déterministe : la formulation change d'un essai à l'autre.

**Question 21.** `.env` contient la clé API : une fois poussée, elle reste dans l'historique Git. `chroma_db_*/` est lourd et se recalcule. `.venv/` est lourd et propre à chaque machine : `requirements.txt` suffit à le recréer. `__pycache__/` contient des fichiers compilés, recréés automatiquement.

**Question 27.** Toute personne qui a le lien consomme les crédits de la clé : un script peut les épuiser, et on peut détourner le chatbot pour un autre usage. Mesures possibles : limiter le nombre de questions par session et leur longueur ; fixer un plafond de dépenses chez le fournisseur ; protéger l'accès par un mot de passe ; utiliser une clé dédiée, facile à révoquer. L'application déployée ici utilise OpenAI, et non Mistral comme le conseille l'énoncé : ces mesures comptent d'autant plus que cette clé est payante.

## Bonus

**Question 28.** Le `.dockerignore` évite de copier dans l'image les secrets, l'historique Git, `.venv/` et les bases. Le `Dockerfile` est écrit, mais non testé.

**Question 30.** Résultats sur 5 questions, évalués par mots-clés : réponse trouvée dans les passages / réponse jugée correcte.

| `chunk_size` | k = 2 | k = 4 | k = 8 |
|---|---|---|---|
| 300 | 4 / 3 | 5 / 5 | 5 / 5 |
| 1000 | 4 / 4 | 5 / 5 | 5 / 4 |
| 2000 | 5 / 5 | 5 / 5 | 5 / 5 |

Le k = 2 trouve moins souvent la réponse avec de petits chunks. Le reste n'est pas interprétable : seulement 5 questions, et un modèle non déterministe. Ce score surestime aussi la qualité : une réponse comme « le contexte ne précise pas, mais mentionne Stability AI » contient le mot-clé.

**Question 31.** MMR donne des passages plus variés : sur les GAN, seuls 2 chunks sur 4 restent communs, et d'autres pages apparaissent. La réponse reste proche. MMR ne résout pas le cas de Stable Diffusion.

**Question 33.** Le modèle ajoute bien des numéros comme [1]. Mais quand il s'abstient, il cite quand même [1][2][3][4] : ces citations ne prouvent donc pas tout.
