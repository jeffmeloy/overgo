import json

items = json.load(open('docs/plan.json')).get('items', [])
ids = {item['id'] for item in items}
wanted = {'session-supervised-training', 'promotion-gate-burndown', 'external-benchmark-breadth'}
missing = wanted - ids
if missing:
    raise SystemExit('circuit campaign rows missing: %s' % sorted(missing))
print('circuit campaign rows present')
