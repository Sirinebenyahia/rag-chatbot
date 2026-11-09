# 💬 Chatbot RAG 100% fonctionnel — LLM gratuit (Groq) + Streamlit

Un chatbot RAG (Retrieval-Augmented Generation) **classique** et **prêt à l'emploi** :
tu uploades tes documents (PDF/TXT), le bot les indexe, et répond à tes questions
en se basant uniquement sur leur contenu.

## 🧱 Stack utilisée (100% gratuite)

| Composant        | Outil                                   | Coût                     |
|-------------------|------------------------------------------|---------------------------|
| LLM (génération)  | **Groq API** (Llama 3.1 8B Instant)       | Gratuit (quota généreux)  |
| Embeddings        | `sentence-transformers/all-MiniLM-L6-v2` | Gratuit, tourne en local  |
| Base vectorielle  | FAISS (en mémoire)                        | Gratuit, local            |
| Interface         | Streamlit                                 | Gratuit, local            |

Aucune carte bancaire n'est requise. Le seul prérequis est une clé API Groq gratuite.

---

## 🚀 Installation

### 1. Cloner / télécharger le projet et se placer dedans

```bash
cd rag-chatbot-groq
```

### 2. Créer un environnement virtuel (recommandé)

```bash
python -m venv venv
source venv/bin/activate      # sur Windows : venv\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Obtenir une clé API Groq gratuite

1. Va sur https://console.groq.com/keys
2. Crée un compte gratuit (email ou Google)
3. Clique sur "Create API Key"
4. Copie la clé (elle commence par `gsk_...`)

### 5. Configurer la clé API

Copie le fichier `.env.example` en `.env` et colle ta clé dedans :

```bash
cp .env.example .env
```

Puis édite `.env` :

```
GROQ_API_KEY=gsk_ta_vraie_cle_ici
```

> Tu peux aussi simplement coller la clé directement dans l'interface Streamlit
> (barre latérale) sans passer par le fichier `.env`.

---

## ▶️ Lancer l'application

```bash
streamlit run app.py
```

L'application s'ouvre automatiquement dans ton navigateur à l'adresse
`http://localhost:8501`.

---

## 🧪 Tester rapidement

Un document d'exemple est fourni dans `sample_docs/exemple_entreprise.txt`.

1. Lance l'application (`streamlit run app.py`)
2. Dans la barre latérale, uploade `sample_docs/exemple_entreprise.txt`
3. Clique sur **"🔨 Construire / mettre à jour l'index"**
4. Pose une question, par exemple :
   - *"Quelle est la politique de remboursement ?"*
   - *"Quels sont les produits de NovaTech ?"*
   - *"Le support est-il disponible le week-end ?"*

Le chatbot doit répondre en citant les extraits pertinents du document.

---

## 📂 Structure du projet

```
rag-chatbot-groq/
├── app.py                       # Application Streamlit (interface + logique RAG)
├── rag_utils.py                 # Fonctions : chargement, découpage, embeddings, FAISS
├── requirements.txt             # Dépendances Python
├── .env.example                 # Exemple de fichier de configuration
├── sample_docs/
│   └── exemple_entreprise.txt   # Document de test
└── README.md
```

---

## 🔍 Comment fonctionne le pipeline RAG

1. **Ingestion** : le(s) document(s) uploadé(s) sont chargés (`PyPDFLoader` / `TextLoader`)
2. **Découpage** : le texte est divisé en chunks de ~1000 caractères avec chevauchement
3. **Embeddings** : chaque chunk est converti en vecteur via un modèle local gratuit
   (`all-MiniLM-L6-v2`), donc aucune donnée n'est envoyée à un service tiers pour cette étape
4. **Indexation** : les vecteurs sont stockés dans un index FAISS en mémoire
5. **Retrieval** : à chaque question, les 4 chunks les plus proches sémantiquement
   sont récupérés
6. **Generation** : les chunks + la question sont envoyés au LLM Groq (Llama 3.1),
   qui génère une réponse basée uniquement sur ce contexte

---

## ⚙️ Personnalisation

- **Changer de modèle Groq** : modifie `GROQ_MODEL` dans `app.py`. Modèles gratuits
  disponibles sur https://console.groq.com/docs/models (ex : `llama-3.3-70b-versatile`
  pour plus de qualité, un peu plus lent).
- **Nombre de chunks récupérés** : modifie `k=4` dans l'appel à `retrieve_relevant_chunks`
  dans `app.py`.
- **Taille des chunks** : modifie `chunk_size` / `chunk_overlap` dans `split_documents`
  (`rag_utils.py`).
- **Formats de fichiers supportés** : actuellement PDF, TXT, MD. Tu peux ajouter
  d'autres loaders LangChain dans `rag_utils.py` (ex : `.docx`, `.csv`).

---

## ❗ Dépannage

- **"Erreur lors de l'appel à l'API Groq"** : vérifie que ta clé API est correcte
  et qu'elle n'a pas expiré. Vérifie aussi ton quota sur https://console.groq.com
- **Le téléchargement du modèle d'embeddings est lent au premier lancement** :
  c'est normal, `all-MiniLM-L6-v2` (~80 Mo) est mis en cache localement après le
  premier téléchargement. Les lancements suivants seront rapides.
- **Erreur d'installation `faiss-cpu`** : assure-toi d'utiliser Python 3.9 à 3.12.

---

## 📜 Licence

Libre d'utilisation et de modification pour tes projets personnels ou professionnels.
