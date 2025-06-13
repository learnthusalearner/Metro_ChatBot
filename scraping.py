import requests
from bs4 import BeautifulSoup
import os
import time
from tqdm import tqdm

# Station codes
station_codes = {
    "IITK": "IIT Kanpur", "KLNM": "Kalyanpur", "SPMH": "SPM Hospital", "VWLM": "Vishwavidyalaya",
    "GDCH": "Gurudev Chauraha", "GTNG": "Geeta Nagar", "RWPM": "Rawatpur", "LLRH": "LLR Hospital",
    "MTJM": "Moti Jheel", "CHGJ": "Chunniganj", "NMKT": "Naveen Market", "BDCH": "Bada Chauraha",
    "NYGJ": "Nayaganj", "KNCM": "Kanpur Central", "JHKT": "Jhakarkati Bus Terminal",
    "TPNG": "Transport Nagar", "BRDV": "Baradevi", "KDWN": "Kidwai Nagar",
    "VSNH": "Vasant Vihar", "BDHN": "Baudh Nagar", "NBST": "Naubasta"
}

base_url = "https://yometro.com/from-{src}-metro-station-kanpur-to-{dest}-metro-station-kanpur"
os.makedirs("data", exist_ok=True)

def slugify(name: str) -> str:
    return name.strip().lower().replace(" ", "-")

def create_summary(data):
    parts = []
    title = data.get("Trip Title", "")
    summary = data.get("Route Summary", "")
    if title:
        parts.append(f"This trip details a journey from {title}.")
    if summary:
        parts.append(summary)

    freq = data.get("Train Frequency", {})
    if freq:
        peak = freq.get("PEAK HOURS", {})
        non_peak = freq.get("NON-PEAK HOURS", {})
        if peak:
            peak_summary = ", ".join(f"{day}: {time}" for day, time in peak.items() if time)
            parts.append(f"Train frequency during peak hours: {peak_summary}.")
        if non_peak:
            non_peak_summary = ", ".join(f"{day}: {time}" for day, time in non_peak.items() if time)
            parts.append(f"Non-peak hours frequency: {non_peak_summary}.")

    amenities = data.get("Available Amenities", [])
    if amenities:
        parts.append(f"Amenities include: {', '.join(amenities)}.")

    station_info = data.get("METRO STATION INFO", {})
    for k, v in station_info.items():
        if isinstance(v, list):
            for promo in v:
                for key, val in promo.items():
                    parts.append(f"{key}: {val}")
        else:
            parts.append(f"{k}: {v}")

    trip_info = data.get("Trip Info", {})
    for k, v in trip_info.items():
        parts.append(f"{k}: {v}")

    return " ".join(parts)

def scrape_trip(src_slug, dest_slug, src_code, dest_code):
    url = base_url.format(src=src_slug, dest=dest_slug)
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200 or "404" in resp.text or "page not found" in resp.text.lower():
            return {
                "Trip Title": f"{station_codes[src_code]} to {station_codes[dest_code]}",
                "Route Summary": f"This trip from {station_codes[src_code]} to {station_codes[dest_code]} is not available. It might be under construction or not in the database."
            }

        soup = BeautifulSoup(resp.text, "html.parser")
        data = {}

        trip_raw = url.rsplit("/", 1)[-1]
        data["Trip Title"] = (
            trip_raw.replace("from-", "")
                    .replace("-metro-station-kanpur", "")
                    .replace("-to-", " to ")
                    .replace("-", " ")
                    .title()
        )

        # Train Frequency
        freq = {"PEAK HOURS": {}, "NON-PEAK HOURS": {}}
        table = soup.find("table", class_="table-frequency")
        if table:
            for row in table.find_all("tr")[1:]:
                cols = [td.text.strip() for td in row.find_all("td")]
                if len(cols) >= 3:
                    freq["PEAK HOURS"][cols[0]] = cols[1]
                    freq["NON-PEAK HOURS"][cols[0]] = cols[2]
        data["Train Frequency"] = freq

        # Amenities
        data["Available Amenities"] = [p.text.strip() for p in soup.select("div.amenities p") if p.text.strip()]

        # Station Info
        info = {}
        for p in soup.select("div.address-details p"):
            strong = p.find("strong")
            if strong:
                key = strong.text.strip().rstrip(":")
                val = p.get_text().replace(strong.text, "").strip().lstrip(":")
                if key and val:
                    info[key] = val
        for label, value in zip(soup.select("div.label"), soup.select("div.value")):
            k = label.text.strip().rstrip(":")
            if k in {"CONTACT NO.", "STATION LAYOUT", "PLATFORM TYPE"}:
                info[k] = value.text.strip()
        promo_info = []
        for promo in soup.select("div.promo"):
            lbl = promo.find("span", class_="label")
            if lbl:
                lt = lbl.text.strip()
                content = promo.get_text(" ", strip=True).replace(lt, "", 1).strip()
                if content:
                    promo_info.append({lt: content})
        info["Promotional Info"] = promo_info
        for sel in ("div.info-item", "div.info-item1"):
            for item in soup.select(sel):
                lbl = item.find("span", class_="label")
                val = item.find("div", class_="value")
                if lbl and val:
                    key = lbl.text.strip()
                    info[key] = val.text.replace("⨝", "").strip()
        data["METRO STATION INFO"] = info

        # Trip Info
        trip_info = {}
        for hs in soup.select("div.horizontal-separator"):
            lbl = hs.find("span", class_="label")
            if lbl and "NETWORK/LINE" in lbl.text:
                val = hs.find("div", class_="value")
                loc = val.find("span", class_="location")
                main = val.text.replace(loc.text, "").strip(" /")
                trip_info["NETWORK/LINE"] = f"{main} / {loc.text.strip()}"
            elif lbl and "Areas around the station" in lbl.text:
                p = hs.find("p")
                if p:
                    trip_info["AREAS AROUND THE STATION"] = p.text.strip()
        data["Trip Info"] = trip_info

        # Route Summary
        accordion = soup.find("div", class_="i-amphtml-accordion-content")
        if accordion:
            for promo in accordion.find_all("div", class_="promo"):
                lbl = promo.find("span", class_="label")
                if lbl and lbl.text.strip().lower() == "route summary":
                    data["Route Summary"] = promo.get_text(" ", strip=True).replace(lbl.text.strip(), "").strip()
                    break

        return data

    except Exception as e:
        return {
            "Trip Title": f"{station_codes[src_code]} to {station_codes[dest_code]}",
            "Route Summary": f"Error retrieving data for trip from {station_codes[src_code]} to {station_codes[dest_code]}: {str(e)}"
        }

# Main execution: all station pairs
output_path = "data/all_trips_summary.txt"
with open(output_path, "w", encoding="utf-8") as f_out:
    for src_code, src_name in tqdm(station_codes.items(), desc="Source Stations"):
        for dest_code, dest_name in station_codes.items():
            if src_code == dest_code:
                continue
            src_slug = slugify(src_name)
            dest_slug = slugify(dest_name)

            trip_data = scrape_trip(src_slug, dest_slug, src_code, dest_code)
            summary = create_summary(trip_data)
            f_out.write(summary + "\n\n")
            time.sleep(0.5)  # be polite to the server
