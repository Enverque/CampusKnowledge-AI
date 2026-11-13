from rag_backend import load_vectorstore

vectorstore = load_vectorstore()

print("=" * 80)
print("SEARCHING FOR 'NORTHERN'")
print("=" * 80)

found = 0

for i, doc in enumerate(vectorstore.docstore._dict.values()):

    text = doc.page_content.lower()

    if "northern" in text:

        found += 1

        print("\n" + "-" * 80)
        print("MATCH:", found)
        print("SOURCE:", doc.metadata.get("source_file"))
        print("PAGE:", doc.metadata.get("page_number"))
        print("CATEGORY:", doc.metadata.get("category"))

        print("\nCONTENT:")
        print(doc.page_content[:2000])

print("\n" + "=" * 80)
print("TOTAL MATCHES:", found)
print("=" * 80)