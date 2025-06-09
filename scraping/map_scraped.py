import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
import json

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Fetch sitemap
url = "https://portal.upmetrorail.com/api/v2/sitemap/"
try:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
except requests.RequestException as e:
    print(f"❌ Error fetching sitemap: {e}")
    exit(1)

# Parse XML with BeautifulSoup
soup = BeautifulSoup(response.content, "xml")

results = []

# Process each <url><loc>
for url_tag in soup.find_all("url"):
    loc_tag = url_tag.find("loc")
    if loc_tag and loc_tag.text.startswith("https://kanpur"):
        full_url = loc_tag.text
        parsed_url = urlparse(full_url)
        query = parse_qs(parsed_url.query)

        from_value = query.get("from", [""])[0]
        to_value = query.get("to", [""])[0]

        results.append({
            "from": from_value,
            "to": to_value,
            "link": full_url
        })

# Save as JSON in data folder
output_path = os.path.join("data", "kanpur_map_links.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"✅ {output_path} created")