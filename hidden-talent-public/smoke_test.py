"""
Hidden Talent - connection smoke test
=====================================
Checks that you can reach every Azure resource your team was given, and
prints one sample record from each data source so you can see the fields.

How to run in Azure Cloud Shell (easiest, already signed in):
  1. Upload this file (Manage files -> Upload)
  2. python3 -m pip install --user azure-identity azure-cosmos azure-search-documents openai azure-storage-blob
  3. python3 smoke_test.py

How to run on your laptop:
  1. az login
  2. az account set --subscription "<your team subscription>"
  3. pip install azure-identity azure-cosmos azure-search-documents openai azure-storage-blob
  4. python smoke_test.py

Locally, sign-in uses your own Azure login (from `az login`).
In Azure (Container App), the same code uses the managed identity
aispire-t01-app instead: set AZURE_CLIENT_ID to that identity's client ID.

No API keys are used anywhere. Sign-in is through Microsoft Entra ID.
"""

import json
import os

from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# --- Settings ---------------------------------------------------------------
# Check these against the portal. You can override any of them with an
# environment variable of the same name instead of editing the file.

# Foundry: copy the endpoint from Foundry portal -> your project ->
# Models + endpoints -> "chat" deployment -> Target URI (the part before /openai/...).
FOUNDRY_ENDPOINT = os.getenv("FOUNDRY_ENDPOINT", "https://aispire-t01-ai-alvmb4wnkyzua.openai.azure.com/")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "2025-04-01-preview")
CHAT_DEPLOYMENT = os.getenv("CHAT_DEPLOYMENT", "chat")
EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYMENT", "embedding")

SEARCH_ENDPOINT = os.getenv("SEARCH_ENDPOINT", "https://aispire-t01-search-alvmb4wnkyzua.search.windows.net")
SEARCH_INDEXES = ["jobs", "taxonomy"]

COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT", "https://aispire-t01-cosmos-alvmb4wnkyzua.documents.azure.com:443/")
COSMOS_DATABASE = os.getenv("COSMOS_DATABASE", "hidden-talent")
COSMOS_CONTAINERS = ["candidates", "matches"]

# The storage account was not in the screenshot. Set its name to run that check.
STORAGE_ACCOUNT = os.getenv("STORAGE_ACCOUNT", "")
STORAGE_CONTAINER = "datasets"

if os.getenv("CONTAINER_APP_NAME"):
    # Running inside Azure Container Apps: use the managed identity.
    credential = DefaultAzureCredential()
else:
    # Laptop or Azure Cloud Shell: use your own `az login` sign-in.
    credential = DefaultAzureCredential(exclude_managed_identity_credential=True)


# --- Helpers ----------------------------------------------------------------

def show(sample, max_chars=1500):
    """Print a record as readable JSON, shortened if it is long."""
    text = json.dumps(sample, indent=2, ensure_ascii=False, default=str)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n  ... (truncated)"
    print(text)


def hide_vectors(record):
    """Replace long number lists (embeddings) with a short note."""
    clean = {}
    for key, value in record.items():
        if key.startswith("@") or key.startswith("_"):
            continue  # skip Search/Cosmos system fields
        if isinstance(value, list) and value and isinstance(value[0], float):
            clean[key] = f"<vector, {len(value)} dims>"
        else:
            clean[key] = value
    return clean


def openai_client():
    from openai import AzureOpenAI

    token_provider = get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default"
    )
    return AzureOpenAI(
        azure_endpoint=FOUNDRY_ENDPOINT,
        azure_ad_token_provider=token_provider,
        api_version=OPENAI_API_VERSION,
    )


# --- Checks -----------------------------------------------------------------

def check_cosmos():
    from azure.cosmos import CosmosClient

    client = CosmosClient(COSMOS_ENDPOINT, credential=credential)
    database = client.get_database_client(COSMOS_DATABASE)
    for name in COSMOS_CONTAINERS:
        container = database.get_container_client(name)
        partition_key = container.read().get("partitionKey", {}).get("paths")
        count = list(container.query_items(
            "SELECT VALUE COUNT(1) FROM c", enable_cross_partition_query=True
        ))[0]
        print(f"Container '{name}': {count} items, partition key {partition_key}")
        if count:
            sample = list(container.query_items(
                "SELECT TOP 1 * FROM c", enable_cross_partition_query=True
            ))[0]
            print(f"Sample record from '{name}':")
            show(hide_vectors(sample))


def check_search_data():
    from azure.search.documents import SearchClient

    for index in SEARCH_INDEXES:
        client = SearchClient(SEARCH_ENDPOINT, index, credential)
        print(f"Index '{index}': {client.get_document_count()} documents")
        sample = next(iter(client.search(search_text="*", top=1)), None)
        if sample:
            print(f"Sample document from '{index}':")
            show(hide_vectors(dict(sample)))


def check_search_schema():
    """Reads index definitions. This also tells you whether your team is
    allowed to create the new 'candidates' index later."""
    from azure.search.documents.indexes import SearchIndexClient

    client = SearchIndexClient(SEARCH_ENDPOINT, credential)
    print("Indexes on the service:", list(client.list_index_names()))
    for index in SEARCH_INDEXES:
        definition = client.get_index(index)
        print(f"Fields in '{index}':")
        for field in definition.fields:
            dims = getattr(field, "vector_search_dimensions", None)
            extra = f" ({dims} dims)" if dims else ""
            print(f"  - {field.name}: {field.type}{extra}")


def check_chat():
    client = openai_client()
    response = client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=[{
            "role": "user",
            "content": "Translate to English in one line: "
                       "'Systemadministratör med erfarenhet av Linux'",
        }],
        max_completion_tokens=500,
        reasoning_effort="low",  # gpt-5-mini is a reasoning model: no temperature
    )
    print("Reply:", response.choices[0].message.content)


def check_embedding():
    client = openai_client()
    response = client.embeddings.create(
        model=EMBEDDING_DEPLOYMENT, input="Linux system administrator"
    )
    print("Embedding length:", len(response.data[0].embedding), "(expected 1536)")


def check_storage():
    if not STORAGE_ACCOUNT:
        print("Skipped: set STORAGE_ACCOUNT to your storage account name to run this.")
        return "SKIPPED"
    from azure.storage.blob import BlobServiceClient

    service = BlobServiceClient(
        f"https://{STORAGE_ACCOUNT}.blob.core.windows.net", credential=credential
    )
    container = service.get_container_client(STORAGE_CONTAINER)
    names = [blob.name for blob in container.list_blobs()]
    print("Files:", names)
    if "README.md" in names:
        with open("README.md", "wb") as fh:
            fh.write(container.download_blob("README.md").readall())
        print("Downloaded README.md into this folder.")


# --- Run everything ---------------------------------------------------------

CHECKS = [
    ("Cosmos DB (candidates, matches)", check_cosmos),
    ("AI Search data (jobs, taxonomy)", check_search_data),
    ("AI Search schema (index fields)", check_search_schema),
    ("Foundry chat model", check_chat),
    ("Foundry embedding model", check_embedding),
    ("Storage (datasets)", check_storage),
]

HINTS = {
    "401": "not signed in: run `az login` and pick your team subscription",
    "403": "signed in, but your account lacks a role on this resource: ask the organisers",
    "404": "name or endpoint is wrong: check it in the portal",
    "DeploymentNotFound": "deployment name is wrong: check Models + endpoints in Foundry",
    "CredentialUnavailable": "no Azure login found: run `az login`",
}


def run_check(name, fn):
    print(f"\n=== {name} ===")
    try:
        status = fn() or "OK"
    except Exception as error:  # keep going so you see every result
        message = f"{type(error).__name__}: {str(error)[:400]}"
        print("ERROR:", message)
        for code, hint in HINTS.items():
            if code in message:
                print("Hint:", hint)
        status = "FAIL"
    return status


if __name__ == "__main__":
    summary = [(name, run_check(name, fn)) for name, fn in CHECKS]
    print("\n=== Summary ===")
    for name, status in summary:
        print(f"[{status:7}] {name}")
