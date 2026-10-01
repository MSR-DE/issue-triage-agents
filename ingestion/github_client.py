import os 
from dotenv import load_dotenv
import requests

load_dotenv()

TOKEN = os.getenv("GITHUB_TOKEN")
if not TOKEN:
    raise RuntimeError("GITHUB_TOKEN missing from .env")

HEADERS = {                                         ## HTTP headers for GitHub API requests
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

def get(url, params=None): 
    response = requests.get(url, headers=HEADERS, params=params, timeout=30)
    try:
        body = response.json() ## parse the response body as JSON
    except ValueError:
        body = {"error": response.text[:500]} ## if parsing fails, use the raw text
    next_url = response.links.get("next", {}).get("url")
    return response.status_code, body, next_url


if __name__ == "__main__":
    status, body, next_url = get(
        "https://api.github.com/repos/astral-sh/uv/issues",
        {"state": "all", "per_page": 5},
    )

    print(status)
    print(body[0]["title"])
    print(next_url)