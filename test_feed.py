import requests, sqlite3

# Login with OTP
requests.post('http://localhost:8000/api/auth/login', data={'username':'sandu123','password':'Admin123!'})
conn = sqlite3.connect('backend/trash_detection.db')
otp = conn.execute("SELECT code FROM otp_codes WHERE user_id=(SELECT id FROM users WHERE username='sandu123') AND is_used=0 ORDER BY id DESC LIMIT 1").fetchone()[0]
conn.close()
r2 = requests.post('http://localhost:8000/api/auth/verify-otp', json={'username':'sandu123','code':otp})
tok = r2.json()['access_token']
h = {'Authorization': f'Bearer {tok}'}

cf = requests.get('http://localhost:8000/api/community/feed?limit=2', headers=h).json()
print("TYPE:", type(cf))
if cf:
    print("KEYS:", list(cf[0].keys()))
    import json
    print("ITEM:", json.dumps(cf[0], indent=2, default=str))
