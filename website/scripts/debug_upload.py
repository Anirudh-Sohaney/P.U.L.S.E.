"""Debug upload error."""
import urllib.request
import urllib.parse
import json
import uuid

BASE = "http://localhost:8000"

# Login
data = urllib.parse.urlencode({"username": "admin", "password": "admin123"}).encode()
req = urllib.request.Request(f"{BASE}/api/auth/token", data=data)
r = urllib.request.urlopen(req)
token = json.loads(r.read())["access_token"]

# Upload
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
try:
    r = urllib.request.urlopen(req)
    print("OK:", r.read().decode())
except urllib.error.HTTPError as e:
    print(f"Status: {e.code}")
    body_err = e.read().decode()
    print(f"Body: {body_err}")
