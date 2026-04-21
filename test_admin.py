import requests, json

base = "http://localhost:8000"
errors = []

def check(label, condition, detail=""):
    if condition:
        print(f"  OK  {label}")
    else:
        print(f"  FAIL {label} {detail}")
        errors.append(label)

# ============ ADMIN FLOW ============
print("=" * 60)
print("ADMIN FLOW (user: admin)")
print("=" * 60)

r = requests.post(f"{base}/api/auth/login", data={"username": "admin", "password": "Admin123!"})
tok = r.json()["access_token"]
ah = {"Authorization": f"Bearer {tok}"}
check("Admin login", r.status_code == 200)

# Profile
p = requests.get(f"{base}/api/me/profile", headers=ah).json()
check("Profile returns username", p["username"] == "admin")
check("Profile has eco_score", "eco_score" in p)
check("Profile has rank", "rank" in p)
check("Profile has streak_days", "streak_days" in p)
check("Profile has privacy fields", "anonymous_reports" in p and "hide_exact_location" in p)
print(f"       -> eco={p['eco_score']} rank={p['rank']} reports={p['total_reports']}")

# Notifications
n = requests.get(f"{base}/api/me/notifications", headers=ah).json()
check("Notifications endpoint", isinstance(n, dict) and "notifications" in n)
print(f"       -> {len(n['notifications'])} total, {n['unread']} unread")

# Admin stats
s = requests.get(f"{base}/api/admin/stats", headers=ah).json()
check("Admin stats", s["total_users"] >= 1)
check("Stats has community fields", all(k in s for k in ["total_votes", "pending_reports", "verified_reports", "cleaned_reports", "fake_reports", "in_progress_reports"]))
print(f"       -> users={s['total_users']} sessions={s['total_sessions']} objects={s['total_objects']}")
print(f"       -> votes={s['total_votes']} pending={s['pending_reports']} verified={s['verified_reports']} cleaned={s['cleaned_reports']}")

# Admin users
u = requests.get(f"{base}/api/admin/users", headers=ah).json()
check("Admin users", len(u) >= 1)
check("Users have eco fields", all(k in u[0] for k in ["eco_score", "rank", "streak_days"]))
print(f"       -> {len(u)} users")
for x in u[:3]:
    print(f"          {x['username']:12} role={x['role']:5} eco={x['eco_score']} rank={x['rank']}")

# Admin reports
r2 = requests.get(f"{base}/api/admin/reports?limit=3", headers=ah).json()
check("Admin reports", r2["total"] >= 0)
if r2["items"]:
    check("Reports have status", "status" in r2["items"][0])
    check("Reports have verification_score", "verification_score" in r2["items"][0])
print(f"       -> {r2['total']} total reports")

# Filter tests
for st in ["pending", "verified", "cleaned", "fake", "in_progress"]:
    r3 = requests.get(f"{base}/api/admin/reports?status={st}&limit=1", headers=ah).json()
    check(f"Filter {st}", isinstance(r3, dict) and "total" in r3, f"got {type(r3)}")

# ============ USER FLOW ============
print()
print("=" * 60)
print("USER FLOW (user: sandu123)")
print("=" * 60)

r = requests.post(f"{base}/api/auth/login", data={"username": "sandu123", "password": "Admin123!"})
check("User login step1 (OTP sent)", r.status_code == 200 and r.json().get("otp_required"))

# Read OTP from DB directly
import sqlite3
conn = sqlite3.connect("backend/trash_detection.db")
cur = conn.execute("SELECT code FROM otp_codes WHERE user_id=(SELECT id FROM users WHERE username='sandu123') AND is_used=0 ORDER BY id DESC LIMIT 1")
otp_code = cur.fetchone()[0]
conn.close()
print(f"       -> OTP from DB: {otp_code}")

r2 = requests.post(f"{base}/api/auth/verify-otp", json={"username": "sandu123", "code": otp_code})
check("User login step2 (OTP verify)", r2.status_code == 200, f"status={r2.status_code} body={r2.text[:200]}")
if r2.status_code == 200:
    utok = r2.json()["access_token"]
    uh = {"Authorization": f"Bearer {utok}"}

    # User profile
    up = requests.get(f"{base}/api/me/profile", headers=uh).json()
    check("User profile", up["username"] == "sandu123")
    print(f"       -> eco={up['eco_score']} rank={up['rank']} reports={up['total_reports']} streak={up['streak_days']}")

    # User notifications
    un = requests.get(f"{base}/api/me/notifications", headers=uh).json()
    check("User notifications", "notifications" in un)
    print(f"       -> {len(un['notifications'])} total, {un['unread']} unread")

    # Community feed
    cf = requests.get(f"{base}/api/community/feed?limit=5", headers=uh).json()
    check("Community feed", isinstance(cf, list))
    print(f"       -> {len(cf)} items in feed")
    if cf:
        check("Feed has session_id", "session_id" in cf[0])
        check("Feed has event_type", "event_type" in cf[0])
        check("Feed has status", "status" in cf[0])
        for x in cf[:2]:
            print(f"          #{x['session_id']} user={x.get('username','anon')} status={x['status']} objs={x['total_objects']}")

    # Leaderboard
    lb = requests.get(f"{base}/api/leaderboard?limit=5", headers=uh).json()
    check("Leaderboard", len(lb) >= 1)
    check("Leaderboard has user_rank", "user_rank" in lb[0])
    for x in lb[:3]:
        print(f"          #{x['rank']} {x['username']} eco={x['eco_score']} rank={x['user_rank']}")

    # Map reports
    mr = requests.get(f"{base}/api/map/reports?limit=5", headers=uh).json()
    check("Map reports", isinstance(mr, list))
    if mr:
        check("Map has GPS", "latitude" in mr[0] and "longitude" in mr[0])
        check("Map has status", "status" in mr[0])

    # Ranks info
    rk = requests.get(f"{base}/api/ranks", headers=uh).json()
    check("Ranks endpoint", isinstance(rk, list) and len(rk) == 6)
    for x in rk[:3]:
        print(f"          {x['name']}: {x['min_score']}-{x.get('max_score','inf')} eco")

    # Stats
    st = requests.get(f"{base}/api/stats", headers=uh).json()
    check("Stats endpoint", "total_sessions" in st)

    # ============ VOTE TEST ============
    print()
    print("=" * 60)
    print("COMMUNITY VOTE TEST")
    print("=" * 60)

    # Find a session not owned by sandu123 to vote on
    # Pick a session with NULL reporter that hasn't been voted on yet
    import sqlite3
    conn2 = sqlite3.connect("backend/trash_detection.db")
    votable = conn2.execute(
        "SELECT id FROM detection_sessions WHERE (reporter_id IS NULL OR reporter_id != 2) "
        "AND id NOT IN (SELECT session_id FROM community_votes WHERE user_id=2) "
        "ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn2.close()
    target_sid = votable[0] if votable else 1
    print(f"       -> Voting on session #{target_sid}")
    
    vote_r = requests.post(f"{base}/api/sessions/{target_sid}/vote", headers=uh, json={"vote_type": "confirm", "comment": "Looks legit!"})
    check("Vote on session", vote_r.status_code == 200, f"status={vote_r.status_code} body={vote_r.text[:200]}")
    if vote_r.status_code == 200:
        vd = vote_r.json()
        print(f"       -> votes_up={vd.get('votes_up')} votes_down={vd.get('votes_down')} score={vd.get('verification_score')}")

    # Try voting again (should fail - duplicate)
    vote_r2 = requests.post(f"{base}/api/sessions/{target_sid}/vote", headers=uh, json={"vote_type": "confirm"})
    check("Duplicate vote rejected", vote_r2.status_code in [400, 409], f"status={vote_r2.status_code}")

    # Try voting on own session (should fail)
    # sandu123 owns sessions 42-46
    own_vote = requests.post(f"{base}/api/sessions/46/vote", headers=uh, json={"vote_type": "confirm"})
    check("Self-vote rejected", own_vote.status_code == 400, f"status={own_vote.status_code}")

    # Check admin notifications after vote
    an = requests.get(f"{base}/api/me/notifications", headers=ah).json()
    print(f"\n  Admin notifications after vote: {len(an['notifications'])} total, {an['unread']} unread")
    if an["notifications"]:
        for ni in an["notifications"][:3]:
            print(f"    -> [{ni['category']}] {ni['message'][:60]}")

    # ============ SETTINGS TEST ============
    print()
    print("=" * 60)
    print("SETTINGS TEST")
    print("=" * 60)

    # Update privacy settings
    sr = requests.patch(f"{base}/api/me/settings", headers=uh, json={"anonymous_reports": True, "hide_exact_location": True})
    check("Update settings", sr.status_code == 200)

    # Verify settings persisted
    up2 = requests.get(f"{base}/api/me/profile", headers=uh).json()
    check("Anonymous reports ON", up2["anonymous_reports"] == True)
    check("Hide location ON", up2["hide_exact_location"] == True)

    # Check community feed privacy
    cf2 = requests.get(f"{base}/api/community/feed?limit=20", headers=ah).json()
    sandu123_in_feed = [x for x in cf2 if x["session_id"] in [42, 43, 44, 45, 46]]
    if sandu123_in_feed:
        for x in sandu123_in_feed[:2]:
            is_anon = x.get("username") in [None, "Anonim"]
            check(f"Session #{x['session_id']} is anonymous", is_anon, f"username={x.get('username')}")

    # Reset settings
    requests.patch(f"{base}/api/me/settings", headers=uh, json={"anonymous_reports": False, "hide_exact_location": False})

# ============ SUMMARY ============
print()
print("=" * 60)
if errors:
    print(f"FAILED: {len(errors)} tests failed:")
    for e in errors:
        print(f"  - {e}")
else:
    print("ALL TESTS PASSED!")
print("=" * 60)
