"""
Shared settings and Azure clients for the Hidden Talent scripts.

Every script imports from here, so endpoints and sign-in live in one place.
Override any setting with an environment variable of the same name.

Sign-in:
  - Laptop / Azure Cloud Shell: your own `az login` account
  - Azure Container Apps: the managed identity (set AZURE_CLIENT_ID to the
    client ID of aispire-t01-app)
No API keys are used.
"""

import os

from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# --- Settings ---------------------------------------------------------------
FOUNDRY_ENDPOINT = os.getenv("FOUNDRY_ENDPOINT", "https://aispire-t01-ai-alvmb4wnkyzua.openai.azure.com/")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "2025-04-01-preview")
CHAT_DEPLOYMENT = os.getenv("CHAT_DEPLOYMENT", "chat")
EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYMENT", "embedding")
EMBEDDING_DIMENSIONS = 1536

SEARCH_ENDPOINT = os.getenv("SEARCH_ENDPOINT", "https://aispire-t01-search-alvmb4wnkyzua.search.windows.net")

COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT", "https://aispire-t01-cosmos-alvmb4wnkyzua.documents.azure.com:443/")
COSMOS_DATABASE = os.getenv("COSMOS_DATABASE", "hidden-talent")

# --- Sign-in ----------------------------------------------------------------
if os.getenv("CONTAINER_APP_NAME"):
    credential = DefaultAzureCredential()  # managed identity in Azure
else:
    credential = DefaultAzureCredential(exclude_managed_identity_credential=True)


# --- Clients ----------------------------------------------------------------
def openai_client():
    from openai import AzureOpenAI

    token_provider = get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default"
    )
    return AzureOpenAI(
        azure_endpoint=FOUNDRY_ENDPOINT,
        azure_ad_token_provider=token_provider,
        api_version=OPENAI_API_VERSION,
        max_retries=5,  # retries automatically on rate limits (429)
    )


def cosmos_container(name):
    from azure.cosmos import CosmosClient

    client = CosmosClient(COSMOS_ENDPOINT, credential=credential)
    return client.get_database_client(COSMOS_DATABASE).get_container_client(name)


def search_client(index_name):
    from azure.search.documents import SearchClient

    return SearchClient(SEARCH_ENDPOINT, index_name, credential)
