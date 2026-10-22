"""CPU verification of a completed fine-tune and its preserved parent actor."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    completed = json.loads((args.results/'completed.json').read_text())
    contract = json.loads((args.results/'training_contract.json').read_text())
    assert completed['training_completed'] is True
    path = Path(completed['checkpoint'])
    assert path.parent.resolve() == args.results.resolve()
    assert digest(path) == completed['checkpoint_sha256']
    assert digest(args.parent) == contract['actor_sha256']
    original_hash = digest(args.original)
    assert original_hash == 'e2bb747acb4737f9e73dc34446cfae692ca2564ed36560fb1f137814eea46d33'
    candidate = torch.load(path,map_location='cpu',weights_only=False)
    parent = torch.load(args.parent,map_location='cpu',weights_only=False)
    state, previous = candidate['model_state_dict'], parent['model_state_dict']
    assert all(bool(torch.isfinite(value).all()) for value in state.values())
    keys = [key for key in state if key.startswith('actor.')]
    assert keys and all(key in previous for key in keys)
    delta = torch.cat([(state[key]-previous[key]).flatten() for key in keys])
    assert float(delta.norm()) > 0.
    std = state['std']
    assert bool((std > 0.).all())
    record = dict(candidate_sha256=digest(path), parent_actor_sha256=digest(args.parent),
        original_actor_sha256=original_hash, original_actor_unchanged=True,
        actor_tensors=len(keys), actor_parameter_delta_l2=float(delta.norm()),
        actor_parameter_max_abs_delta=float(delta.abs().max()), all_policy_parameters_finite=True,
        std_min=float(std.min()),std_max=float(std.max()), checkpoint_iteration=completed['last_iteration'],
        updates=completed['updates'])
    args.output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
