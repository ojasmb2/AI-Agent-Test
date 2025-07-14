import os
from dotenv import load_dotenv
from supabase import create_client, Client
from langchain_community.embeddings import HuggingFaceEmbeddings
from uuid import uuid4

# ----------------------------
# 1. Load your environment
# ----------------------------
load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ----------------------------
# 2. Setup your embeddings model
# ----------------------------
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# ----------------------------
# 3. Load your text data
# ----------------------------
# Example: a single string
docs = [
    {
        "content": "The sky is blue and clear today.",
        "metadata": {"source": "example"}
    },
    {
        "content": "Quantum computing uses qubits instead of bits.",
        "metadata": {"source": "example"}
    }
]

# ----------------------------
# 4. Embed and insert each chunk
# ----------------------------
for doc in docs:
    text = doc["content"]
    metadata = doc["metadata"]

    embedding = embedder.embed_query(text)

    # Ensure the vector dimension matches your DB!
    if len(embedding) != 768:
        raise ValueError(f"Embedding dimension {len(embedding)} does not match your table vector(768)!")

    # Insert into Supabase
    data = {
        "id": str(uuid4()),
        "content": text,
        "embedding": embedding,
        "metadata": metadata
    }

    response = supabase.table("documents").insert(data).execute()
    print("Inserted:", response)

print("✅ All chunks embedded and uploaded!")
