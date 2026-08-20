import json

items = json.load(open('docs/plan.json')).get('items', [])
ids = {item['id'] for item in items}
wanted = {'wan-liveedit-closures', 'eval-tool-upgrade', 'memory-bounded-fullstack'}
missing = wanted - ids
if missing:
    raise SystemExit('video campaign rows missing: %s' % sorted(missing))
print('video campaign rows present')
