import os
import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

station_codes = {
    "IITK": "IIT Kanpur",
    "KLNM": "Kalyanpur",
    "SPMH": "SPM HOSPITAL",
    "VWLM": "VISHWAVIDYALAYA METRO",
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

BASE_URL = "https://portal.upmetrorail.com/en/api/v2/travel_distance_time_fare/2"
HEADERS = {
    'User-Agent': 'Mozilla/5.0'
}

def fetch_fare(from_code, to_code):
    url = f"{BASE_URL}/{from_code}/{to_code}/"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()

        total_time = data.get("total_time") or data.get("time") or "0:00:00"
        weekday_fare = data.get("weekday_fare") or data.get("fare") or 0
        stations_count = data.get("stations", 0)

        route_path = [
            {"name": station_codes[from_code], "status": ""},
            {"name": station_codes[to_code], "status": ""}
        ]

        return {
            "status": "success",
            "stations": stations_count,
            "from": station_codes[from_code],
            "to": station_codes[to_code],
            "total_time": total_time,
            "weekday_fare": weekday_fare,
            "weekend_fare": 0,
            "route": [{
                "line": "#3e77bc",
                "line_no": 3,
                "path": route_path,
                "path_time": total_time,
                "map-path": [f"{from_code}-{to_code}"],
                "station_interchange_time": 0,
                "start": station_codes[from_code],
                "end": station_codes[to_code],
                "towards_station": "",
                "direction": ""
            }]
        }

    except Exception as e:
        return {
            "status": "failed",
            "from": station_codes[from_code],
            "to": station_codes[to_code],
            "error": str(e)
        }

def scrape_all_fares():
    results = []

    station_pairs = [
        (from_code, to_code)
        for from_code in station_codes
        for to_code in station_codes
        if from_code != to_code
    ]

    print(f"🔄 Total pairs: {len(station_pairs)}")
    pbar = tqdm(total=len(station_pairs), desc="Scraping", unit="pair")

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(fetch_fare, f, t): (f, t)
            for f, t in station_pairs
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            pbar.update(1)
            time.sleep(0.1)  # polite delay

    pbar.close()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    file_name = f"data/kanpur_fares_formatted_{timestamp}.json"  # Modified path

    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done! Saved to {file_name}")
    success = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    print(f"✅ Successful: {success} | ❌ Failed: {failed}")

if __name__ == "__main__":
    print("🚇 Starting Kanpur Metro Fare Scraper (Formatted Output)...")
    scrape_all_fares()