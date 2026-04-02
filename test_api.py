import urllib.request, json

base = 'http://127.0.0.1:8000'
results = {}

# Session detail (the 500 bug)
try:
    r = urllib.request.urlopen(base+'/api/sessions/46', timeout=5)
    d = json.loads(r.read())
    results['session/46'] = 'OK id=%s records=%s' % (d['id'], len(d['records']))
except urllib.error.HTTPError as e:
    results['session/46'] = 'HTTP %s' % e.code
except Exception as e:
    results['session/46'] = 'ERR %s' % e

# Sessions page
try:
    r = urllib.request.urlopen(base+'/api/sessions?limit=3', timeout=5)
    d = json.loads(r.read())
    results['sessions?limit=3'] = 'OK total=%s items=%s' % (d['total'], len(d['items']))
except Exception as e:
    results['sessions?limit=3'] = 'ERR %s' % e

# Stats
try:
    r = urllib.request.urlopen(base+'/api/stats', timeout=5)
    d = json.loads(r.read())
    results['stats'] = 'OK sessions=%s objects=%s' % (d['total_sessions'], d['total_objects'])
except Exception as e:
    results['stats'] = 'ERR %s' % e

# Leaderboard
try:
    r = urllib.request.urlopen(base+'/api/leaderboard', timeout=5)
    d = json.loads(r.read())
    results['leaderboard'] = 'OK %s users' % len(d)
except Exception as e:
    results['leaderboard'] = 'ERR %s' % e

# Map reports filtered
try:
    r = urllib.request.urlopen(base+'/api/map/reports?resolved=0&material=metal', timeout=5)
    d = json.loads(r.read())
    results['map?resolved=0&material=metal'] = 'OK %s reports' % len(d)
except Exception as e:
    results['map?resolved=0&material=metal'] = 'ERR %s' % e

# Zones
try:
    r = urllib.request.urlopen(base+'/api/zones', timeout=5)
    d = json.loads(r.read())
    results['zones'] = 'OK %s zones' % len(d)
except Exception as e:
    results['zones'] = 'ERR %s' % e

# CSV export
try:
    r = urllib.request.urlopen(base+'/api/export/csv', timeout=6)
    lines = r.read().decode('utf-8').splitlines()
    results['csv'] = 'OK header=%s rows=%s' % (lines[0][:40], len(lines)-1)
except Exception as e:
    results['csv'] = 'ERR %s' % e

for k, v in results.items():
    ok = 'PASS' if v.startswith('OK') else 'FAIL'
    print('%s  %-35s %s' % (ok, k, v))
