from rag_backend import load_vectorstore


vectorstore = load_vectorstore()

query = "placement information for students in the Northern region"

results = vectorstore.similarity_search(
    query,
    k=10
)

print("\n" + "=" * 80)
print("SEARCH RESULTS")
print("=" * 80)

for i, doc in enumerate(results, start=1):

    print(f"\nRESULT {i}")
    print("-" * 80)

    print("SOURCE:", doc.metadata.get("source_file"))
    print("PAGE:", doc.metadata.get("page_number"))
    print("CATEGORY:", doc.metadata.get("category"))

    print("\nCONTENT:")
    print(doc.page_content[:1500])
    