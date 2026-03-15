"""Collect visual review agent results into a findings JSON keyed by filename."""
import json
import re
import sys

all_findings = []

for path in sys.argv[1:]:
    with open(path) as f:
        # Each line is a JSONL entry; parse to get the text content
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Get text content from the message
            msg = entry.get('message', {})
            content = msg.get('content', '')
            if isinstance(content, list):
                # Content blocks
                for block in content:
                    if isinstance(block, dict) and block.get('type') == 'text':
                        content = block.get('text', '')
                        break
                else:
                    continue

            if not isinstance(content, str) or '"file"' not in content:
                continue

            # Find JSON arrays in the text
            for match in re.finditer(r'\[\s*\{', content):
                start = match.start()
                depth = 0
                for i in range(start, len(content)):
                    if content[i] == '[':
                        depth += 1
                    elif content[i] == ']':
                        depth -= 1
                        if depth == 0:
                            candidate = content[start:i+1]
                            try:
                                items = json.loads(candidate)
                                if (isinstance(items, list) and items
                                        and isinstance(items[0], dict)
                                        and 'file' in items[0]):
                                    all_findings.extend(items)
                            except json.JSONDecodeError:
                                pass
                            break

# Filter out positive/non-actionable findings
positive_words = ['good', 'correct', 'well-framed', 'no issues', 'acceptable',
                  'charming', 'clean crop', 'valid', 'no issue']
filtered = []
for f in all_findings:
    desc_lower = f['desc'].lower()
    # Skip if the description is purely positive
    if any(desc_lower.strip().endswith(w) for w in ['good.', 'correct.', 'good']):
        continue
    if desc_lower.startswith('good') or desc_lower.startswith('correct'):
        continue
    if 'no issues' in desc_lower and 'crop_validity' not in desc_lower.lower():
        continue
    filtered.append(f)

# Deduplicate
seen = set()
unique = []
for f in filtered:
    key = (f['file'], f['desc'][:50])
    if key not in seen:
        seen.add(key)
        unique.append(f)

# Group by filename
by_file = {}
for f in unique:
    fname = f['file']
    if fname not in by_file:
        by_file[fname] = []
    by_file[fname].append({'type': f['type'], 'desc': f['desc']})

output = '/tmp/visual_review/findings.json'
with open(output, 'w') as f:
    json.dump(by_file, f, indent=2)

print(f"Collected {len(unique)} findings for {len(by_file)} files -> {output}")
