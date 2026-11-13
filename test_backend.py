from rag_backend import load_vectorstore

vectorstore = load_vectorstore()

print("FAISS loaded successfully!")
print("Total vectors:", vectorstore.index.ntotal)