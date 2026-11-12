import base64
import os
from typing import Any, Dict, List, Union

import dotenv
import requests
from langchain_core.documents import Document

dotenv.load_dotenv()

GITHUB_API_URL = "https://api.github.com"
GITHUB_OWNER = os.getenv("GITHUB_OWNER")
GITHUB_REPO = os.getenv("GITHUB_REPO")
GITHUB_DIR_PATH = os.getenv("GITHUB_DIR_PATH", "")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")


def validate_github_env() -> None:
    """Validate that all required GitHub environment variables are present."""
    missing = [
        var_name
        for var_name, value in [
            ("GITHUB_OWNER", GITHUB_OWNER),
            ("GITHUB_REPO", GITHUB_REPO),
            ("GITHUB_TOKEN", GITHUB_TOKEN),
        ]
        if not value
    ]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")


def _github_api_request(endpoint: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Helper method to make authenticated requests to the GitHub API."""
    validate_github_env()

    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }
    url = f"{GITHUB_API_URL}/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents/{endpoint.lstrip('/')}"
    
    response = requests.get(url, headers=headers, timeout=10)
    if response.status_code != 200:
        raise RuntimeError(
            f"GitHub API Error [{response.status_code}]: {response.text}"
        )
    return response.json()


def fetch_github_directory_contents(dir_path: str = GITHUB_DIR_PATH) -> List[Dict[str, Any]]:
    """Fetch directory metadata list from GitHub repository."""
    contents = _github_api_request(dir_path)
    if not isinstance(contents, list):
        raise TypeError(f"Expected a directory list at '{dir_path}', got a single file object instead.")
    return contents


def fetch_github_file_content(file_path: str) -> str:
    """Fetch and decode plain text content of a file from GitHub repository."""
    file_data = _github_api_request(file_path)
    
    if isinstance(file_data, list):
        raise TypeError(f"Expected a file path at '{file_path}', but found a directory.")
    
    encoded_content = file_data.get("content", "")
    decoded_bytes = base64.b64decode(encoded_content)
    return decoded_bytes.decode("utf-8", errors="ignore")

def load_github_documents(dir_path: str = GITHUB_DIR_PATH) -> List[Document]:
    """Load documents from a GitHub directory into a list of Document objects."""
    pages = fetch_github_directory_contents(dir_path)
    documents = []

    for page in pages:
        if page.get("type") == "file":
            content = fetch_github_file_content(page["path"])
            documents.append(Document(
                page_content=content,
                metadata={"source": page["name"], "path": page["path"]}
            ))

    return documents

if __name__ == "__main__":
    try:
        documents = load_github_documents()
        print(f"Loaded {len(documents)} document(s) from '{GITHUB_DIR_PATH}' in '{GITHUB_OWNER}/{GITHUB_REPO}'.\n")

        for doc in documents:
            print(f"Document Name: {doc.metadata['source']}\nContent Preview:\n{doc.page_content[:200]}...\n")
    except (ValueError, RuntimeError, TypeError) as err:
        print(f"Error: {err}")