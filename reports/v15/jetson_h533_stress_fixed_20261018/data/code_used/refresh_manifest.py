"""Accept only explicitly reviewed latency-fix payloads and preserve the baseline."""
import hashlib
import json
from pathlib import Path

root = Path('/home/scienceowl/robot_dog/cpp_20261017/v15_jetson_20261012')
here = Path(__file__).resolve().parent
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
manifest = root / 'SHA256SUMS'
before = manifest.read_text()
provenance = json.loads((here / 'latency_fix_provenance.json').read_text())
assert digest(manifest) == provenance['baseline_manifest_sha256']
old = {name: value for value, name in (line.split('  ', 1) for line in before.splitlines())}
changes = provenance['files']
for name, expected in old.items():
    if name in changes:
        assert expected == changes[name]['before_sha256'], name
    else:
        assert digest(root / name) == expected, 'Unrelated payload changed: ' + name
for name, item in changes.items():
    assert old.get(name) == item['before_sha256'], name
    assert digest(root / name) == item['after_sha256'], 'New source mismatch: ' + name
with (root / 'SHA256SUMS.before_cuda_graph_20261018').open('x') as stream:
    stream.write(before)
note = root / 'LATENCY_FIX_20261018.json'
note.write_text(json.dumps(provenance, indent=2) + '\n')
old.update({name: item['after_sha256'] for name, item in changes.items()})
old[note.name] = digest(note)
temp = root / 'SHA256SUMS.latency.tmp'
temp.write_text(''.join(value + '  ' + name + '\n' for name, value in sorted(old.items())))
temp.replace(manifest)
print('Preserved baseline and documented', len(changes), 'reviewed files;', len(old), 'payload checksums')
