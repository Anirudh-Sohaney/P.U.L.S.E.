"""Quick test script to verify the full pipeline works."""
import urllib.request
import urllib.parse
import json
import uuid

BASE = "http://localhost:8000"

# 1. Login
print("1. Logging in...")
data = urllib.parse.urlencode({"username": "admin", "password": "admin123"}).encode()
req = urllib.request.Request(f"{BASE}/api/auth/token", data=data)
r = urllib.request.urlopen(req)
token = json.loads(r.read())["access_token"]
print(f"   OK - token: {token[:20]}...")

# 2. Upload CSV
print("2. Uploading CSV...")
with open("data/pharmacy_risk_data.csv", "rb") as f:
    csv_data = f.read()

boundary = uuid.uuid4().hex
body = (
    b"--" + boundary.encode() + b"\r\n"
    b'Content-Disposition: form-data; name="file"; filename="pharmacy_risk_data.csv"\r\n'
    b"Content-Type: text/csv\r\n\r\n"
    + csv_data + b"\r\n"
    b"--" + boundary.encode() + b"--\r\n"
)

req = urllib.request.Request(
    f"{BASE}/api/data/ingest",
    data=body,
    headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    },
    method="POST",
)
r = urllib.request.urlopen(req)
result = json.loads(r.read())
print(f"   OK - {result}")

# 3. Check dashboard
print("3. Checking dashboard...")
req = urllib.request.Request(
    f"{BASE}/api/predictions/dashboard",
    headers={"Authorization": f"Bearer {token}"},
)
r = urllib.request.urlopen(req)
dash = json.loads(r.read())
print(f"   Stats: {dash['stats']}")
print(f"   Top risk meds: {len(dash['top_risk_medications'])}")

# 4. Check predictions
print("4. Checking predictions...")
req = urllib.request.Request(
    f"{BASE}/api/predictions/all",
    headers={"Authorization": f"Bearer {token}"},
)
r = urllib.request.urlopen(req)
preds = json.loads(r.read())
print(f"   Total predictions: {preds['count']}")

# 5. Check model info
print("5. Checking model info...")
req = urllib.request.Request(
    f"{BASE}/api/model/info",
    headers={"Authorization": f"Bearer {token}"},
)
r = urllib.request.urlopen(req)
info = json.loads(r.read())
print(f"   Metrics: {info.get('metrics', 'N/A')}")

print("\nAll tests passed!")
