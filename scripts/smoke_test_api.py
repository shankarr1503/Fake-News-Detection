"""Manual smoke test against a running API (not collected by pytest)."""

import json
import sys

import requests

URL = "http://localhost:8000/predict"
DATA = {
    "title": "Breaking: AI solves world hunger",
    "text": "Scientists have developed an AI that can produce food out of thin air.",
}

try:
    response = requests.post(URL, json=DATA, timeout=60)
except requests.RequestException as e:
    sys.exit(f"Error: {e}")

print(f"Status Code: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")
if not response.ok:
    sys.exit(1)
