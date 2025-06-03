import time
import qrcode
from pymongo import MongoClient
from smartcard.System import readers
from smartcard.util import toHexString
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access environment variables
MONGO_URI = os.getenv("MONGO_URI")
REGISTRATION_SITE = os.getenv("REGISTRATION_SITE")

# MongoDB connection
client = MongoClient(MONGO_URI)
db = client["nfc_project"]
collection = db["card_uids"]

# Track seen cards
current_cards = set()

def read_uid():
    try:
        r = readers()
        if not r:
            return None
        reader = r[0]
        conn = reader.createConnection()
        conn.connect()
        GET_UID = [0xFF, 0xCA, 0x00, 0x00, 0x00]
        data, sw1, sw2 = conn.transmit(GET_UID)
        if sw1 == 0x90:
            return ''.join(format(x, '02X') for x in data)
        return None
    except Exception:
        return None

def generate_qr(url):
    qr = qrcode.make(url)
    qr.show()

print("🟢 Ready for card scanning...")

while True:
    uid = read_uid()

    if uid:
        if uid not in current_cards:
            print(f"🔍 Detected new card: {uid}")
            user = collection.find_one({"uid": uid})

            if user:
                print(f"✅ Card already registered to {user.get('name', 'Unknown')}")
            else:
                print("🆕 New card detected. Generating QR code...")
                registration_url = REGISTRATION_SITE + uid
                generate_qr(registration_url)
            
            current_cards.add(uid)
    else:
        if current_cards:
            print("🟡 Card removed. Waiting for next scan...")
        current_cards.clear()

    time.sleep(1)
