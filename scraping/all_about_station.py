import os
import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Station codes mapping - moved to global scope
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

BASE_URL = "https://portal.upmetrorail.com/en/api/v2/station_detail/2"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def fetch_station_details(station_code):
    """Fetch detailed information for a station"""
    url = f"{BASE_URL}/{station_code}/"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") and data.get("data"):
            return {
                "station_code": station_code,
                "station_name": station_codes.get(station_code, ""),
                "details": data["data"],
                "status": "success"
            }
        return {
            "station_code": station_code,
            "error": "Station not found or invalid response",
            "status": "failed"
        }
    except Exception as e:
        return {
            "station_code": station_code,
            "error": str(e),
            "status": "failed"
        }

def scrape_all_stations():
    results = []
    failed_attempts = []
    
    print(f"Scraping details for {len(station_codes)} stations...")
    
    # Use threading for faster scraping
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(fetch_station_details, code): code
            for code in station_codes
        }
        
        # Initialize progress bar
        pbar = tqdm(total=len(station_codes), desc="🚇 Scraping stations", unit="station")
        
        for future in as_completed(futures):
            result = future.result()
            if result["status"] == "success":
                results.append(result)
            else:
                failed_attempts.append(result)
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'Success': len(results),
                'Failed': len(failed_attempts)
            })
            
            time.sleep(0.3)  # Polite delay
    
    pbar.close()
    
    # Create structured output
    structured_data = {}
    for result in results:
        if result["status"] == "success":
            station_code = result["station_code"]
            details = result["details"]
            
            structured_data[station_code] = {
                "name": details.get("st_name", ""),
                "code": details.get("st_code", ""),
                "type": details.get("st_type", ""),
                "description": details.get("st_desc", ""),
                "line": details.get("metro_lines", [{}])[0].get("ln_name", "") if details.get("metro_lines") else "",
                "status": details.get("st_status", []),
                "latitude": details.get("latitude", ""),
                "longitude": details.get("longitude", ""),
                "contact": {
                    "mobile": details.get("mobile", ""),
                    "landline": details.get("landline", "")
                },
                "facilities": [facility["kind"] for facility in details.get("station_facilities", [])],
                "platforms": [
                    {
                        "name": platform.get("platform_name", ""),
                        "code": platform.get("platform_code", ""),
                        "towards": platform.get("train_towards", {}).get("st_name", "") if platform.get("train_towards") else ""
                    }
                    for platform in details.get("platforms", [])
                ],
                "gates": [
                    {
                        "name": gate.get("gate_name", ""),
                        "code": gate.get("gate_code", ""),
                        "location": gate.get("location", ""),
                        "status": gate.get("status", "")
                    }
                    for gate in details.get("gates", [])
                ],
                "parking": [
                    {
                        "location": park.get("location", ""),
                        "car_capacity": park.get("capacity_car", 0),
                        "motorcycle_capacity": park.get("capacity_motorcycle", 0),
                        "bicycle_capacity": park.get("capacity_cycle", 0)
                    }
                    for park in details.get("parkings", [])
                ],
                "lifts": [
                    {
                        "type": lift.get("lift_type", ""),
                        "name": lift.get("name", ""),
                        "description": lift.get("description_location", "")
                    }
                    for lift in details.get("lifts", [])
                ]
            }
    
    # Save results in data folder
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    success_file = os.path.join("data", f"kanpur_station_details_{timestamp}.json")
    structured_file = os.path.join("data", f"kanpur_stations_structured_{timestamp}.json")
    
    with open(success_file, "w") as f:
        json.dump(results, f, indent=2)
    
    with open(structured_file, "w") as f:
        json.dump(structured_data, f, indent=2)
    
    # Print final report
    print("\n" + "="*50)
    print("📝 Scraping Report".center(50))
    print("="*50)
    print(f"✅ Successful scrapes: {len(results)}/{len(station_codes)}")
    print(f"⚠️ Failed scrapes: {len(failed_attempts)}")
    print(f"💾 Raw results saved to: {success_file}")
    print(f"💾 Structured data saved to: {structured_file}")
    
    # Show sample data
    if structured_data:
        first_station = next(iter(structured_data.values()))
        print("\nSample structured data:")
        print(json.dumps(first_station, indent=2))

if __name__ == "__main__":
    print("Starting Kanpur Metro Station Details Scraper...")
    scrape_all_stations()