import os
from PIL import Image
import streamlit as st

# Dictionary mapping station codes to names
station_codes = {
    "IITK": "IIT Kanpur", "KLMN": "Kalyanpur", "SPMH": "SPM Hospital", "VWLM": "Vishwavidyalaya",
    "GDCH": "Gurudev Chauraha", "GTNG": "Geeta Nagar", "RWPM": "Rawatpur", "LLRH": "LLR Hospital",
    "MTJM": "Moti Jheel", "CHGJ": "Chunniganj", "NMKT": "Naveen Market", "BDCH": "Bada Chauraha",
    "NYGJ": "Nayaganj", "KNCM": "Kanpur Central", "JHKT": "Jhakarkati Bus Terminal",
    "TPNG": "Transport Nagar", "BRDV": "Baradevi", "KDWN": "Kidwai Nagar",
    "VSNH": "Vasant Vihar", "BDHN": "Baudh Nagar", "NBST": "Naubasta"
}


def get_station_codes():
    return station_codes


def resolve_station_code(input_str: str) -> str:
    """
    Resolves a user-provided code or full name to a valid station code.

    :param input_str: Code or full name (case-insensitive)
    :return: Matching station code or None
    """
    input_str = input_str.strip().lower()
    for code, name in station_codes.items():
        if input_str == code.lower() or input_str == name.lower():
            return code
    return None


def display_connection_image(code1: str, code2: str, folder_path: str = "loc_images") -> None:
    
    code1 = resolve_station_code(code1)
    code2 = resolve_station_code(code2)

    if not code1 or not code2:
        st.warning("❌ Invalid station name or code provided.")
        return

    filenames = [f"{code1}-{code2}.png", f"{code2}-{code1}.png"]
    for filename in filenames:
        image_path = os.path.join(folder_path, filename)
        if os.path.exists(image_path):
            st.image(Image.open(image_path), caption=f"{station_codes[code1]} ↔ {station_codes[code2]}")
            return

    st.warning("⚠️ No image found for the selected station pair.")
