# Hadoop Incident Triage (POC)

POC Streamlit pour analyser des **logs publics Hadoop (HDFS / YARN)** avec un **LLM local (Ollama)**.

Objectifs principaux :

- Charger des logs Hadoop (HDFS / YARN) publics
- Calculer des **signaux rapides** (heuristiques simples mais utiles)
- Construire une **timeline** (bursts d'événements par minute)
- Faire une **recherche RAG** (FAISS) pour extraire des extraits pertinents
- Appliquer une logique **multi‑agents** :
  1. Agent HDFS : expert HDFS (replication, DataNode, IO, NameNode)
  2. Agent YARN : expert YARN (RM/NM, containers, ressources, blacklisting)
  3. Agent Reviewer : « principal SRE » qui challenge et consolide
- Exporter un **rapport RCA** en Markdown

> 🔒 **Compliance** : tout est local (LLM via Ollama) et les logs utilisés doivent être **publics**.

---

## 1. Prérequis

- Python 3.9+ installé sur la machine
- [Git](https://git-scm.com/) + Git Bash (sous Windows)
- [Ollama](https://ollama.com/) installé et démarré
- Accès au modèle Ollama **`qwen2.5:1.5b`**

Sous Windows, le projet est pensé pour fonctionner via **Git Bash** (par exemple dans Wave Terminal).

Les jeux de logs Hadoop utilisés pour le POC peuvent, par exemple, être extraits depuis le projet **[Loghub](https://github.com/logpai/loghub)** au format texte brut. Un script d’aide est fourni dans le répertoire `scripts/` pour automatiser le téléchargement / l’extraction de certains jeux de logs publics.

---

## 2. Installation

Clone du repo puis installation locale :

```bash
git clone https://github.com/TON_ORG/hadoop-incident-triage.git
cd hadoop-incident-triage
```

### 2.1. Créer l’environnement virtuel

Sous Git Bash / Linux / macOS :

```bash
python3 -m venv .venv
```

> Sur Windows (Git Bash), l’exécutable Python du venv est dans `./.venv/Scripts/python.exe`.

### 2.2. Activer le venv et installer les dépendances

Sous Git Bash :

```bash
source .venv/Scripts/activate   # Windows + Git Bash
# source .venv/bin/activate     # Linux / macOS

python -m pip install -U pip
python -m pip install -r requirements.txt
```

### 2.3. Télécharger le modèle Ollama

Assure‑toi qu’Ollama est démarré, puis :

```bash
ollama pull qwen2.5:1.5b
```

---

## 3. Lancer l’application Streamlit

Avec le venv **activé** :

```bash
streamlit run app.py
```

Ou, si tu préfères appeler explicitement l’exécutable du venv :

```bash
.venv/Scripts/streamlit.exe run app.py      # Windows + Git Bash
# .venv/bin/streamlit run app.py           # Linux / macOS
```

Streamlit t’indiquera l’URL locale, typiquement :

- http://localhost:8501

---

## 4. Utilisation de l’UI

### 4.1. Réglages (barre latérale)

- **Modèle Ollama** : par défaut `qwen2.5:1.5b`
- **Chunk size** : taille des chunks pour le RAG (ex. 600 caractères)
- **Chunk overlap** : recouvrement entre chunks
- **TopK RAG** : nombre d’extraits les plus pertinents à récupérer
- **Limite chars / fichier (RAM)** : coupe les fichiers trop volumineux pour rester léger

### 4.2. Étapes d’analyse

1. **Charger des logs**  
   - Drag & drop un ou plusieurs fichiers `.log`, `.txt`, `.md` (HDFS / YARN, logs publics).
2. **Signaux rapides + timeline**  
   - L’app calcule :
     - nombre total de lignes
     - mots‑clés fréquents (`error`, `warn`, `exception`, `replica`, `container`, etc.)
     - « composants » naïfs (token avant `:` ou `[pid]`)
   - Une **timeline par minute** est affichée si des timestamps au format `YYYY-MM-DD HH:MM` sont détectés.
3. **Analyse LLM (multi‑agents)**  
   - Bouton **« Lancer l’analyse »** :
     - Indexation RAG des logs dans FAISS
     - Agent HDFS : analyse des symptômes côté HDFS
     - Agent YARN : analyse des symptômes côté YARN
     - Agent Reviewer : produit une **RCA consolidée** en français
4. **Export du rapport**  
   - Bouton **« Télécharger RCA_HADOOP_POC.md »** : export Markdown contenant :
     - signaux rapides (JSON)
     - timeline (JSON)
     - analyse HDFS
     - analyse YARN
     - RCA consolidée

---

## 5. Architecture technique

- **Frontend** : [Streamlit](https://streamlit.io/)
- **LLM** : [Ollama](https://ollama.com/) + modèle local `qwen2.5:1.5b`
- **RAG** :
  - Découpage des logs via `RecursiveCharacterTextSplitter`
  - Vectorisation avec `FastEmbedEmbeddings` (modèle `BAAI/bge-small-en-v1.5`)
  - Indexation / recherche avec **FAISS** (`langchain_community.vectorstores.FAISS`)
- **Multi‑agents** :
  - 3 appels LLM avec des prompts spécialisés (HDFS, YARN, Reviewer)

---

## 6. Limitations & notes

- POC uniquement : pas prévu pour de la prod telle‑quelle.
- Les heuristiques de **signaux rapides** et de **timeline** sont volontairement simples.
- Les logs doivent être **anonymisés / publics**. Ne pas utiliser de données sensibles.
- Performances et qualité de l’analyse dépendent :
  - du modèle local (taille, capacité de raisonnement),
  - de la quantité/taille des logs fournis.

---

## 7. Dépannage (FAQ rapide)

### `ModuleNotFoundError: No module named 'langchain_text_splitters'`

Vérifie que les dépendances sont bien installées dans le bon venv :

```bash
source .venv/Scripts/activate
python -m pip install -r requirements.txt
```

### `ollama: command not found` ou problème de modèle

- Vérifie qu’Ollama est installé et lancé.
- Vérifie que `ollama` est dans le `PATH`.
- Vérifie que le modèle `qwen2.5:1.5b` est bien présent :

  ```bash
  ollama list
  ```

---

## 8. Licence

À compléter avec la licence souhaitée (MIT, Apache 2.0, etc.).
