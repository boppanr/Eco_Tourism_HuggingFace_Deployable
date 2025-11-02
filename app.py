import gradio as gr
import pickle
import faiss
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from transformers import pipeline

# ---------------------------------------------------------
# STEP 1: Load FAISS index and metadata
# ---------------------------------------------------------
print("🔹 Loading dataset...")

with open("eco_tourism_meta.pkl", "rb") as f:
    meta_data = pickle.load(f)

if isinstance(meta_data, list):
    print(f"✅ Loaded metadata list with {len(meta_data)} entries")
else:
    print("⚠️ Metadata format not list; proceeding anyway")

print("🔹 Loading FAISS index...")
index = faiss.read_index("index.faiss")

with open("index.pkl", "rb") as f:
    store_data = pickle.load(f)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

docs = []
for d in meta_data:
    if isinstance(d, str):
        docs.append(Document(page_content=d))
    elif isinstance(d, dict) and "page_content" in d:
        docs.append(Document(page_content=d["page_content"]))
    else:
        docs.append(Document(page_content=str(d)))

docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(docs)})
index_to_docstore_id = {i: str(i) for i in range(len(docs))}

vectorstore = FAISS(
    embedding_function=embedding_model,
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id,
)
print("✅ Vector store ready!")

# ---------------------------------------------------------
# STEP 2: Initialize small lightweight LLM
# ---------------------------------------------------------
print("🔹 Initializing lightweight FLAN-T5 model...")

llm_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",  # lightweight & CPU friendly
    max_new_tokens=256,
)
print("✅ FLAN-T5 ready!")


# ---------------------------------------------------------
# STEP 3: Define query function
# ---------------------------------------------------------
def answer_query(query):
    try:
        results = vectorstore.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in results])

        prompt = f"""
You are an eco-tourism expert. Use the context below to answer the question accurately.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""
        response = llm_pipeline(prompt)[0]["generated_text"]
        return response.strip()
    except Exception as e:
        return f"⚠️ Error: {str(e)}"


# ---------------------------------------------------------
# STEP 4: Gradio Web UI
# ---------------------------------------------------------
def chat_interface(message, history):
    response = answer_query(message)
    history.append((message, response))
    return history, history


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🌿 EcoTourism AI Assistant (Lightweight)")
    gr.Markdown("Ask about sustainable travel, eco-destinations, or green tourism tips!")

    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(label="Your question:")
    clear = gr.Button("Clear Chat")

    msg.submit(chat_interface, [msg, chatbot], [chatbot, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch(server_name="0.0.0.0", server_port=7860)
