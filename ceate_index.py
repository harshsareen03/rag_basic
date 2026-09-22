
def create_chunks(text, chunk_size=200):

    chunks = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)

    return chunks


with open("data/company_policy.txt", "r") as file:
    text = file.read()

chunks = create_chunks(text)

for i, chunk in enumerate(chunks):
    print(f"\nCHUNK {i}")
    print(chunk)