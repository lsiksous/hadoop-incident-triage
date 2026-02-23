#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
POC Streamlit — Hadoop Incident Triage (logs publics)

BUT:
- Charger des logs Hadoop publics (HDFS / YARN)
- Construire des "signaux rapides" (heuristiques cheap mais utiles)
- Construire une timeline (bursts d'événements par minute)
- Faire une recherche RAG (FAISS) pour extraire des extraits pertinents
- Appliquer une logique "multi-agents" en 3 rôles:
    1) Agent HDFS: expert HDFS (replication, DataNode, IO, NameNode)
    2) Agent YARN: expert YARN (RM/NM, containers, ressources, blacklisting)
    3) Agent Reviewer: "principal SRE" qui challenge et consolide
- Exporter un rapport RCA Markdown

IMPORTANT:
- Tout est local: LLM via Ollama (ex: qwen2.5:1.5b)
- Aucun fichier sensible: logs publics uniquement
"""

import io
import json
import re
from collections import Counter, defaultdict
from typing import List, Tuple

import streamlit as st
import matplotlib.pyplot as plt

from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


# -----------------------------------------------------------------------------
# 1) Parsing timeline (bursts)
# -----------------------------------------------------------------------------
# On cherche un timestamp simple "YYYY-MM-DD HH:MM" dans chaque ligne.
# Pour un POC c'est volontairement "bête", mais ça marche souvent sur des logs Hadoop.
TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2})")


def build_timeline(text: str) -> dict:
    """
    Construit une timeline "événements/minute".
    BUT: mettre en évidence des bursts (pics de logs) -> utile pour montrer un incident.
    """
    timeline = defaultdict(int)
    for line in text.splitlines():
        m = TS_RE.search(line)
        if m:
            timeline[m.group(1)] += 1
    return dict(sorted(timeline.items()))


# -----------------------------------------------------------------------------
# 2) Signaux rapides (cheap heuristics)
# -----------------------------------------------------------------------------
# Avant même d'appeler le LLM, on extrait des signaux:
# - nombre total de lignes
# - mots-clés fréquents (error/warn/replica/container...)
# - "composants" naïfs (1er token ou token avant ':')
#
# BUT:
# - offrir une "base factuelle" rapide
# - stabiliser le LLM (moins d'hallucinations) en lui donnant des chiffres
def naive_signals(text: str) -> dict:
    keywords = [
        "error", "warn", "exception", "failed",
        "block", "replica", "heartbeat",
        "namenode", "datanode",
        "disk", "ioexception", "lease",
        "safe mode", "under-replicated",
        "container", "resourcemanager", "nodemanager",
        "blacklist", "killed", "timeout",
    ]
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    kw_counts = Counter()
    comp_counts = Counter()

    for l in lines:
        lo = l.lower()
        for k in keywords:
            if k in lo:
                kw_counts[k] += 1

        # heuristique "composant":
        # - si la ligne commence par "xxx:" -> composant xxx
        # - ou "xxx[pid]" -> composant xxx
        token = l.split()[0] if l.split() else ""
        if ":" in token:
            comp_counts[token.split(":")[0]] += 1
        elif "[" in token and "]" in token:
            comp_counts[token.split("[")[0]] += 1

    return {
        "lines_total": len(lines),
        "keyword_hits": kw_counts.most_common(10),
        "top_components": comp_counts.most_common(10),
    }


# -----------------------------------------------------------------------------
# 3) Préparation RAG (docs -> vectorstore FAISS)
# -----------------------------------------------------------------------------
# Ici le RAG n'est pas "chat avec base documentaire" complet,
# c'est juste un mécanisme pour:
# - découper les logs en chunks
# - indexer
# - récupérer les extraits les plus pertinents
#
# Ensuite, on donne ces extraits aux agents (HDFS/YARN/Reviewer).
def make_documents(files: List[Tuple[str, str]]) -> List[Document]:
    return [Document(page_content=content, metadata={"path": name}) for name, content in files]


def build_vectorstore(docs: List[Document], chunk_size: int, chunk_overlap: int) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)

    # Embeddings légers (M1/8Go friendly)
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return FAISS.from_documents(chunks, embeddings)


def rag_search(vs: FAISS, query: str, k: int) -> List[dict]:
    """
    Retourne k extraits les plus proches de la requête.
    On ne donne pas tout le log au LLM (trop long),
    on donne seulement les "meilleurs" extraits.
    """
    results = vs.similarity_search(query, k=k)
    return [{"path": r.metadata.get("path"), "snippet": r.page_content[:900]} for r in results]


# -----------------------------------------------------------------------------
# 4) Visualisation timeline
# -----------------------------------------------------------------------------
def plot_timeline(timeline: dict, max_points: int = 120):
    items = list(timeline.items())
    if len(items) > max_points:
        items = items[-max_points:]

    xs = [k for k, _ in items]
    ys = [v for _, v in items]

    fig = plt.figure()
    plt.plot(range(len(xs)), ys)
    plt.xticks(range(len(xs)), xs, rotation=70, fontsize=8)
    plt.tight_layout()
    return fig


# =============================================================================
# UI Streamlit
# =============================================================================
st.set_page_config(page_title="Hadoop Incident Triage (POC)", layout="wide")
st.title("Hadoop Incident Triage (POC) — Logs publics, LLM local")

with st.sidebar:
    st.header("Réglages")
    model = st.text_input("Modèle Ollama", value="qwen2.5:1.5b")
    chunk_size = st.slider("Chunk size", 300, 1200, 600, 50)
    chunk_overlap = st.slider("Chunk overlap", 0, 300, 80, 10)
    topk = st.slider("TopK RAG", 2, 12, 6, 1)
    max_chars_per_file = st.slider("Limite chars / fichier (RAM)", 2000, 200000, 40000, 1000)
    st.caption("Conseil M1/8Go : modèle 1.5B + chunks ~600 + limite chars ~40k.")

st.subheader("1) Charger des logs Hadoop (publics)")
uploaded = st.file_uploader(
    "Dépose un ou plusieurs fichiers .log/.txt/.md (HDFS/YARN)",
    type=["log", "txt", "md"],
    accept_multiple_files=True,
)

# On charge les fichiers en mémoire (limite pour éviter de saturer le Mac)
files = []
if uploaded:
    for uf in uploaded:
        raw = uf.getvalue()
        txt = raw.decode("utf-8", errors="ignore")
        txt = txt[:max_chars_per_file]
        files.append((uf.name, txt))

colA, colB = st.columns([1, 1])

with colA:
    st.subheader("2) Signaux rapides + timeline")
    if files:
        combined = "\n".join(c for _, c in files)
        signals = naive_signals(combined)
        timeline = build_timeline(combined)

        st.markdown("**Signaux rapides (JSON)**")
        st.json(signals)

        if timeline:
            st.markdown("**Timeline (événements / minute)**")
            st.pyplot(plot_timeline(timeline))
        else:
            st.info("Aucun timestamp reconnu (format attendu: YYYY-MM-DD HH:MM).")
    else:
        st.info("Charge des fichiers pour calculer les signaux.")

with colB:
    st.subheader("3) Analyse LLM (multi-agents) + Export Markdown")
    run = st.button("Lancer l'analyse", type="primary", disabled=not bool(files))

    if run:
        combined = "\n".join(c for _, c in files)
        signals = naive_signals(combined)
        timeline = build_timeline(combined)

        # ---------------------------------------------------------------------
        # A) Indexation RAG
        # ---------------------------------------------------------------------
        # On fabrique un vectorstore FAISS local et on récupère TOPK extraits.
        # Ces extraits servent de "contexte commun" aux agents.
        with st.status("Indexation (RAG) en cours…", expanded=False) as status:
            docs = make_documents(files)
            vs = build_vectorstore(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            # Query volontairement large: on veut attraper symptômes HDFS + YARN
            contexts = rag_search(
                vs,
                "Hadoop incident symptoms root cause HDFS YARN failure replication datanode namenode resourcemanager nodemanager",
                k=topk,
            )
            status.update(label="Indexation terminée.", state="complete")

        # ---------------------------------------------------------------------
        # B) LLM local (Ollama)
        # ---------------------------------------------------------------------
        llm = Ollama(model=model, temperature=0)

        # ---------------------------------------------------------------------
        # C) Multi-agents (LOGIQUE CENTRALE)
        # ---------------------------------------------------------------------
        #
        # Ici on simule une équipe SRE:
        # - Agent HDFS: spécialiste HDFS
        # - Agent YARN: spécialiste YARN
        # - Reviewer: rôle "principal SRE" qui challenge et tranche
        #
        # Pourquoi c'est utile ?
        # - On évite le "monolithe" : un seul prompt qui mélange tout
        # - Chaque agent est meilleur dans son domaine (spécialisation)
        # - Le reviewer réduit le bruit / hallucinations et priorise
        #
        # Important: ce n'est PAS magique: ce sont 3 appels LLM,
        # mais avec des rôles + objectifs différents.
        with st.status("Agents: HDFS / YARN / Reviewer…", expanded=True) as status:

            # ---- Agent 1: HDFS expert ----
            # Il doit:
            # - extraire symptômes HDFS
            # - proposer hypothèses
            # - donner checks (hdfs fsck, dfsadmin -report...)
            # - citer des preuves (snippets)
            hdfs_prompt = f"""
You are a senior HDFS SRE.
Analyze the logs for HDFS issues (replication, DataNode, IO, NameNode, safe mode).
Produce hypotheses + verification steps. Quote short evidence.

Quick signals:
{json.dumps(signals, ensure_ascii=False, indent=2)}

Timeline (may be empty):
{json.dumps(timeline, ensure_ascii=False, indent=2)}

Relevant snippets:
{json.dumps(contexts, ensure_ascii=False, indent=2)}
""".strip()
            hdfs_analysis = llm.invoke(hdfs_prompt)

            # ---- Agent 2: YARN expert ----
            # Même logique, mais domaine YARN:
            # - containers killed / failures
            # - ressources (memory/vcores)
            # - RM/NM / blacklisting
            yarn_prompt = f"""
You are a senior YARN SRE.
Analyze the logs for YARN issues (RM/NM, containers, resources, blacklisting).
Produce hypotheses + verification steps. Quote short evidence.

Quick signals:
{json.dumps(signals, ensure_ascii=False, indent=2)}

Timeline (may be empty):
{json.dumps(timeline, ensure_ascii=False, indent=2)}

Relevant snippets:
{json.dumps(contexts, ensure_ascii=False, indent=2)}
""".strip()
            yarn_analysis = llm.invoke(yarn_prompt)

            # ---- Agent 3: Reviewer / principal SRE ----
            # Rôle:
            # - lire les 2 analyses
            # - challenger les hypothèses faibles
            # - produire une RCA finale "actionnable"
            # - ajouter un score de confiance
            #
            # C'est le reviewer qui "fait la synthèse" et rend un livrable.
            review_prompt = f"""
You are a principal SRE.
Write a concise RCA in French:
1) Symptômes factuels
2) Hypothèses classées + confiance (0-100)
3) Preuves (citations courtes)
4) Vérifications recommandées (commandes Hadoop génériques)
5) Actions immédiates vs structurelles
Remove weak hypotheses.

HDFS analysis:
{hdfs_analysis}

YARN analysis:
{yarn_analysis}
""".strip()
            final_rca = llm.invoke(review_prompt)

            status.update(label="Analyse terminée.", state="complete")

        # ---------------------------------------------------------------------
        # D) Affichage + export Markdown
        # ---------------------------------------------------------------------
        st.markdown("### RCA consolidée")
        st.markdown(final_rca)

        with st.expander("Analyse HDFS", expanded=False):
            st.markdown(hdfs_analysis)

        with st.expander("Analyse YARN", expanded=False):
            st.markdown(yarn_analysis)

        md = io.StringIO()
        md.write("# RCA Hadoop – POC (logs publics)\n\n")
        md.write("## Signaux rapides\n```json\n")
        md.write(json.dumps(signals, ensure_ascii=False, indent=2))
        md.write("\n```\n\n")

        md.write("## Timeline (événements par minute)\n```json\n")
        md.write(json.dumps(timeline, ensure_ascii=False, indent=2))
        md.write("\n```\n\n")

        md.write("## Analyse HDFS\n")
        md.write(hdfs_analysis + "\n\n")

        md.write("## Analyse YARN\n")
        md.write(yarn_analysis + "\n\n")

        md.write("## RCA consolidée\n")
        md.write(final_rca + "\n")

        st.download_button(
            "Télécharger RCA_HADOOP_POC.md",
            data=md.getvalue().encode("utf-8"),
            file_name="RCA_HADOOP_POC.md",
            mime="text/markdown",
        )

st.caption("LLM local via Ollama + modèle Qwen. Logs publics uniquement (POC compliant).")