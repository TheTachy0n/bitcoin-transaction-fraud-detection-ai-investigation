# ============================================================
# STEP 14 — RAG RETRIEVER
# ELLIPTIC BITCOIN FRAUD DETECTION
# ============================================================

from pathlib import Path
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

KNOWLEDGE_BASE_PATH = (
    PROJECT_ROOT / "knowledge_base"
)


# ============================================================
# RAG RETRIEVER
# ============================================================

class RAGRetriever:

    def __init__(self):

        print("Initializing RAG Retriever...")

        self.documents = []

        self._load_documents()

        self._build_index()

        print(
            "Documents loaded:",
            len(self.documents)
        )

        print("RAG Retriever ready.")


    # ========================================================
    # LOAD DOCUMENTS
    # ========================================================

    def _load_documents(self):

        if not KNOWLEDGE_BASE_PATH.exists():

            raise FileNotFoundError(
                f"Knowledge base not found: "
                f"{KNOWLEDGE_BASE_PATH}"
            )


        files = sorted(
            KNOWLEDGE_BASE_PATH.glob("*.txt")
        )


        if not files:

            raise FileNotFoundError(
                "No .txt files found in knowledge_base/"
            )


        print(
            f"Found {len(files)} knowledge files."
        )


        for path in files:

            text = path.read_text(
                encoding="utf-8"
            ).strip()


            # ------------------------------------------------
            # Skip empty files
            # ------------------------------------------------

            if not text:

                print(
                    f"WARNING: {path.name} is empty."
                )

                continue


            print(
                f"Loaded: {path.name} "
                f"({len(text)} characters)"
            )


            self.documents.append({

                "source":
                    path.name,

                "text":
                    text
            })


        if not self.documents:

            raise ValueError(
                "All knowledge-base files are empty."
            )


    # ========================================================
    # BUILD TF-IDF INDEX
    # ========================================================

    def _build_index(self):

        texts = [

            document["text"]

            for document in self.documents
        ]


        print(
            "\nBuilding TF-IDF index..."
        )


        # ----------------------------------------------------
        # IMPORTANT
        #
        # Do NOT remove English stop words.
        #
        # The knowledge base is small and contains important
        # phrases such as:
        #
        # "high risk"
        # "unknown transaction"
        # "fraud investigation"
        #
        # Keeping all words makes retrieval more robust.
        # ----------------------------------------------------

        self.vectorizer = TfidfVectorizer(

            lowercase=True,

            ngram_range=(1, 2),

            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z_]+\b"
        )


        try:

            self.document_matrix = (
                self.vectorizer.fit_transform(
                    texts
                )
            )

        except ValueError as error:

            print(
                "\nERROR: Could not build TF-IDF index."
            )

            print(
                "Document previews:"
            )

            for document in self.documents:

                print(
                    "\n---",
                    document["source"],
                    "---"
                )

                print(
                    repr(
                        document["text"][:300]
                    )
                )

            raise error


        print(
            "Vocabulary size:",
            len(
                self.vectorizer.vocabulary_
            )
        )

        print(
            "Document matrix shape:",
            self.document_matrix.shape
        )


    # ========================================================
    # RETRIEVE
    # ========================================================

    def retrieve(
        self,
        query,
        top_k=3
    ):

        if not query or not query.strip():

            return []


        query_vector = (
            self.vectorizer.transform(
                [query]
            )
        )


        similarities = cosine_similarity(

            query_vector,

            self.document_matrix

        )[0]


        ranked_indices = (
            similarities.argsort()[::-1]
        )


        results = []


        for index in ranked_indices[:top_k]:

            results.append({

                "source":
                    self.documents[index][
                        "source"
                    ],

                "score":
                    float(
                        similarities[index]
                    ),

                "text":
                    self.documents[index][
                        "text"
                    ]
            })


        return results


    # ========================================================
    # FORMAT CONTEXT
    # ========================================================

    def format_context(
        self,
        results
    ):

        if not results:

            return (
                "No relevant evidence found."
            )


        context_parts = []


        for result in results:

            context_parts.append(

                f"SOURCE: {result['source']}\n"
                f"RELEVANCE: "
                f"{result['score']:.4f}\n"
                f"{result['text']}"
            )


        return "\n\n".join(
            context_parts
        )


# ============================================================
# DEMO
# ============================================================

def main():

    print("=" * 70)
    print("STEP 14 — RAG RETRIEVER")
    print("=" * 70)


    retriever = RAGRetriever()


    queries = [

        (
            "What does it mean when "
            "XGBoost and GraphSAGE both "
            "assign high risk?"
        ),

        (
            "How should unknown graph "
            "neighbors be interpreted?"
        ),

        (
            "What do SHAP values mean?"
        ),

        (
            "What should investigators do "
            "with a high risk transaction?"
        )
    ]


    for query in queries:

        print(
            "\n" + "=" * 70
        )

        print(
            "QUERY:"
        )

        print(
            query
        )


        results = retriever.retrieve(
            query,
            top_k=2
        )


        print(
            "\nRETRIEVED EVIDENCE:"
        )


        for result in results:

            print(
                "\nSource:",
                result["source"]
            )

            print(
                "Score:",
                round(
                    result["score"],
                    4
                )
            )

            print(
                "Preview:"
            )

            print(
                result["text"][:500]
            )


    print(
        "\n" + "=" * 70
    )

    print(
        "STEP 14 COMPLETE"
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()