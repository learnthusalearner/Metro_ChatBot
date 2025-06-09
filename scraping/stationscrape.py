# kanpur_metro_scraper.py

import os
import requests as r
import json

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Dictionary of station codes mapped to station names (Line 3 - Kanpur)
station_codes = {
    "IITK": "IIT Kanpur",
    "KLNM": "Kalyanpur",
    "SPMH": "SPM Hospital",
    "VWLM": "Vishwavidyalaya",
    "GDCH": "Gurudev Chauraha",
    "GTNG": "Geeta Nagar",
    "RWPM": "Rawatpur",
    "LLRH": "LLR Hospital",
    "MTJM": "Moti Jheel",
    "CHGJ": "Chunniganj",
    "NMKT": "Naveen Market",
    "BDCH": "Bada Chauraha",
    "NYGJ": "Nayaganj",
    "KNCM": "Kanpur Central",
    "JHKT": "Jhakarkati Bus Terminal",
    "TPNG": "Transport Nagar",
    "BRDV": "Baradevi",
    "KDWN": "Kidwai Nagar",
    "VSNH": "Vasant Vihar",
    "BDHN": "Baudh Nagar",
    "NBST": "Naubasta"
}

base_url = "https://portal.upmetrorail.com/en/api/v2/station_brief_detail/2/"
station_data = {}

for code, name in station_codes.items():
    url = f"{base_url}{code}/"
    print(f"Fetching: {name} ({code}) → {url}")
    try:
        response = r.get(url, timeout=10)
        response.raise_for_status()
        station_data[name] = response.json()
    except r.RequestException as e:
        print(f"❌ Error fetching {name}: {e}")
        station_data[name] = {"error": str(e)}

# Save results to a JSON file in the data folder
output_path = os.path.join("data", "kanpur_metro_line3_status.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(station_data, f, ensure_ascii=False, indent=2)

print(f"✅ Data saved to {output_path}")