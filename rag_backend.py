import os
import re

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace


# ============================================================
# SETTINGS
# ============================================================

INDEX_PATH = "college_faiss_index"


# ============================================================
# LOAD FAISS VECTORSTORE
# ============================================================

def load_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


# ============================================================
# LOAD DEEPSEEK MODEL
# ============================================================

def load_chat_model():

    llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-V3",
        max_new_tokens=512,
        temperature=0.2,
        huggingfacehub_api_token=os.getenv("HF_TOKEN")
    )

    chat_model = ChatHuggingFace(
        llm=llm
    )

    return chat_model


# ============================================================
# KEYWORD SEARCH
# ============================================================

def get_keyword_scores(query, vectorstore):
    """
    Search all indexed chunks using exact keywords.

    Useful for retrieving:
    - regions
    - college names
    - company names
    - years
    - placement information
    - rankings
    - percentages
    - table information
    """

    query = query.lower().strip()

    # Extract words from question
    query_words = re.findall(
        r"\b[a-zA-Z0-9][a-zA-Z0-9\-]*\b",
        query
    )

    # Common words that are not useful for retrieval
    stop_words = {
        "what",
        "which",
        "where",
        "when",
        "who",
        "how",
        "many",
        "much",
        "is",
        "are",
        "was",
        "were",
        "the",
        "a",
        "an",
        "for",
        "of",
        "in",
        "on",
        "to",
        "and",
        "or",
        "with",
        "tell",
        "me",
        "about",
        "please",
        "can",
        "could",
        "would",
        "does",
        "do",
        "did",
        "show",
        "give",
        "provide",
        "information"
    }

    important_words = [
        word
        for word in query_words
        if word not in stop_words and len(word) >= 3
    ]

    # --------------------------------------------------------
    # Detect phrases such as:
    #
    # Northern region
    # Southern region
    # Eastern region
    # Western region
    #
    # This is dynamic. It is NOT hard-coded to Northern.
    # --------------------------------------------------------

    region_phrases = re.findall(
        r"\b([a-zA-Z][a-zA-Z\-]*)\s+region\b",
        query
    )

    scored_documents = []

    # Search every indexed chunk
    for doc in vectorstore.docstore._dict.values():

        text = doc.page_content.lower()

        score = 0

        # ----------------------------------------------------
        # Exact keyword matching
        # ----------------------------------------------------

        for word in important_words:

            matches = re.findall(
                r"\b" + re.escape(word) + r"\b",
                text
            )

            # Each exact keyword occurrence gets points
            score += len(matches) * 2

        # ----------------------------------------------------
        # Region phrase matching
        # ----------------------------------------------------

        for phrase in region_phrases:

            full_phrase = phrase + " region"

            # Strongest boost:
            # "northern region" exactly appears
            if full_phrase in text:

                score += 50

            # If only "northern" appears
            elif phrase in text:

                score += 20

        # ----------------------------------------------------
        # General region boost
        # ----------------------------------------------------

        if "region" in query and "region" in text:

            score += 5

        # ----------------------------------------------------
        # Placement boost
        # ----------------------------------------------------

        if "placement" in query and "placement" in text:

            score += 10

        # ----------------------------------------------------
        # Save document if relevant
        # ----------------------------------------------------

        if score > 0:

            scored_documents.append(
                (score, doc)
            )

    # Highest score first
    scored_documents.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return scored_documents


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(
    query,
    vectorstore,
    semantic_k=10,
    keyword_k=15
):
    """
    Combines:

    1. Semantic FAISS search
    2. Exact keyword search

    This improves retrieval for region-wise information,
    tables, names, years and statistics.
    """

    # --------------------------------------------------------
    # SEMANTIC SEARCH
    # --------------------------------------------------------

    semantic_documents = vectorstore.similarity_search(
        query,
        k=semantic_k
    )

    # --------------------------------------------------------
    # KEYWORD SEARCH
    # --------------------------------------------------------

    keyword_results = get_keyword_scores(
        query,
        vectorstore
    )

    keyword_documents = [
        doc
        for score, doc in keyword_results[:keyword_k]
    ]

    # --------------------------------------------------------
    # COMBINE RESULTS
    # --------------------------------------------------------

    combined_documents = []

    seen = set()

    for doc in semantic_documents + keyword_documents:

        # Create a unique ID for each chunk
        doc_id = (
            doc.metadata.get(
                "source_file",
                ""
            ),
            doc.metadata.get(
                "page_number",
                ""
            ),
            doc.page_content[:150]
        )

        if doc_id not in seen:

            combined_documents.append(doc)

            seen.add(doc_id)

    return combined_documents


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    vectorstore,
    chat_model
):

    # --------------------------------------------------------
    # HYBRID RETRIEVAL
    # --------------------------------------------------------

    documents = hybrid_search(
        question,
        vectorstore,
        semantic_k=10,
        keyword_k=15
    )

    # --------------------------------------------------------
    # DEBUG INFORMATION
    #
    # This helps us see what the RAG system retrieved.
    # You can remove this later.
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("HYBRID RETRIEVAL RESULTS")
    print("=" * 80)

    for i, doc in enumerate(
        documents,
        start=1
    ):

        print(
            f"\nRESULT {i}"
        )

        print(
            "-" * 80
        )

        print(
            "SOURCE:",
            doc.metadata.get(
                "source_file",
                "Unknown"
            )
        )

        print(
            "PAGE:",
            doc.metadata.get(
                "page_number",
                "Unknown"
            )
        )

        print(
            "CATEGORY:",
            doc.metadata.get(
                "category",
                "Unknown"
            )
        )

        print(
            "\nCONTENT:"
        )

        print(
            doc.page_content[:800]
        )

    # --------------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------------

    context_parts = []

    for doc in documents:

        source = doc.metadata.get(
            "source_file",
            "Unknown document"
        )

        page = doc.metadata.get(
            "page_number",
            "Unknown"
        )

        category = doc.metadata.get(
            "category",
            "Unknown"
        )

        context_parts.append(
            f"""
Source: {source}
Page: {page}
Category: {category}

Content:
{doc.page_content}
"""
        )

    context = "\n\n---\n\n".join(
        context_parts
    )

    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are a College Document Assistant.

Answer the student's question ONLY using the
college documents provided in the context.

Do not use outside knowledge.

IMPORTANT RULES:

1. Pay close attention to specific terms in the
student's question, including:

- region
- college name
- company name
- year
- placement
- percentage
- ranking
- fees
- admission
- scholarship
- hostel
- attendance

2. If the student mentions a specific region,
answer using information for THAT region.

3. NEVER replace one region with another region.

4. The student may ask about ANY region mentioned
in the college documents.

5. Do not assume that the region means the location
of the college.

6. Use the region information actually present
inside the documents.

7. If the student asks for ALL regions, provide
the relevant information for all regions available
in the retrieved context.

8. If the student asks for a comparison between
regions, compare only information available in
the documents.

9. If the documents contain tables, carefully
interpret the table information from the context.

10. Preserve numerical values accurately.

11. Do not invent information.

12. Do not use outside knowledge.

13. If the requested information is not available
in the provided context, say:

"I could not find this information in the college documents."

14. Keep the answer clear and student-friendly.

15. Whenever possible, mention the source document
and page number.

CONTEXT:
{context}

STUDENT QUESTION:
{question}

ANSWER:
"""

    # --------------------------------------------------------
    # CALL DEEPSEEK
    # --------------------------------------------------------

    response = chat_model.invoke(
        prompt
    )

    return response.content, documents