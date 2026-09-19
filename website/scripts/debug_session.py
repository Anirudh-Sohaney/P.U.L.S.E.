"""Test: login, upload, then hit dashboard multiple times to simulate session."""
import urllib.request
import urllib.parse
import json
import uuid

BASE = "http://localhost:8000/api"

def api(method, path, token=None, data=None, content_type=None):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if content_type:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(f"{BASE}{path}", data=data, headers=headers, method=method)
    try:
        r = urllib.request.urlopen(req)
        return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": e.code, "body": e.read().decode()}

# 1. Login
print("=== Step 1: Login ===")
login_data = urllib.parse.urlencode({"username": "admin", "password": "admin123"}).encode()
result = api("POST", "/auth/token", data=login_data, content_type="application/x-www-form-urlencoded")
token = result.get("access_token")
print(f"Token: {token[:20]}..." if token else f"FAIL: {result}")

# 2. Check status (should have data from previous upload)
print("\n=== Step 2: Data status ===")
status = api("GET", "/data/status", token=token)
print(json.dumps(status, indent=2))

# 3. Upload NEW dataset
print("\n=== Step 3: Upload new dataset ===")
with open("data/pharmacy_risk_data_crisis.csv", "rb") as f:
    csv_data = f.read()

boundary = uuid.uuid4().hex
body = (
    b"--" + boundary.encode() + b"\r\n"
    b'Content-Disposition: form-data; name="file"; filename="pharmacy_risk_data_crisis.csv"\r\n'
    b"Content-Type: text/csv\r\n\r\n"
    + csv_data + b"\r\n"
    b"--" + boundary.encode() + b"--\r\n"
)

upload_result = api("POST", "/data/ingest", token=token, data=body,
                     content_type=f"multipart/form-data; boundary={boundary}")
print(json.dumps(upload_result, indent=2))

# 4. Dashboard (immediately after upload)
print("\n=== Step 4: Dashboard ===")
dash = api("GET", "/predictions/dashboard", token=token)
if "error" in dash:
    print(f"FAIL: {dash}")
else:
    print(f"Stats: {dash['stats']}")
    print(f"Top risk meds count: {len(dash['top_risk_medications'])}")

# 5. Dashboard again (simulates page refresh)
print("\n=== Step 5: Dashboard (second call) ===")
dash2 = api("GET", "/predictions/dashboard", token=token)
if "error" in dash2:
    print(f"FAIL: {dash2}")
else:
    print(f"Stats: {dash2['stats']}")

# 6. All predictions
print("\n=== Step 6: All predictions ===")
preds = api("GET", "/predictions/all", token=token)
if "error" in preds:
    print(f"FAIL: {preds}")
else:
    print(f"Count: {preds['count']}")
