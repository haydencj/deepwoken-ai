import mwclient
import json
from dataclasses import dataclass

# Define a simple Document class
@dataclass
class Document:
    name: str
    content: str
    metadata: dict

def scrape_deepwoken_wiki():
    # Connect to the Deepwoken Wiki API (using the correct path '/' for Fandom)
    site = mwclient.Site('deepwoken.fandom.com', path='/')
    documents = []

    # Iterate over all pages in the wiki
    for page in site.allpages():
        title = page.name
        content = page.text()  # Retrieve wikitext content
        # Construct a URL for the page (assuming standard Fandom URL pattern)
        url = f"https://deepwoken.fandom.com/wiki/{title.replace(' ', '_')}"
        doc = Document(
            name=title,
            content=content,
            metadata={"url": url, "source": "web"}
        )
        documents.append(doc)
        print(f"Scraped: {title}")

    return documents

def save_documents_to_jsonl(documents, filename="docs.jsonl"):
    with open(filename, "w", encoding="utf-8") as f:
        for doc in documents:
            record = {
                "name": doc.name,
                "content": doc.content,
                "metadata": doc.metadata
            }
            f.write(json.dumps(record) + "\n")
    print(f"Saved {len(documents)} documents to {filename}")

if __name__ == "__main__":
    docs = scrape_deepwoken_wiki()
    save_documents_to_jsonl(docs)
