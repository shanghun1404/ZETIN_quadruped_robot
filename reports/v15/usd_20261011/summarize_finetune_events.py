"""CPU-only export of actual TensorBoard scalars from a CAD training run."""
import argparse
import json
import math
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    reader = EventAccumulator(str(args.results), size_guidance={'scalars': 0})
    reader.Reload()
    scalars = {}
    for tag in reader.Tags()['scalars']:
        events = reader.Scalars(tag)
        assert all(math.isfinite(event.value) for event in events), tag
        values = [event.value for event in events]
        scalars[tag] = dict(count=len(events), first=values[0], last=values[-1],
            first_step=events[0].step, last_step=events[-1].step,
            minimum=min(values), maximum=max(values),
            first50_mean=sum(values[:50])/len(values[:50]),
            last50_mean=sum(values[-50:])/len(values[-50:]),
            events=[[event.step,event.value] for event in events])
    record = dict(schema=1, all_scalar_values_finite=True, scalars=scalars)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    selected = ('Train/mean_reward','Train/mean_episode_length',
        'Curriculum/terrain_levels','Diagnostics/nonfinite_steps',
        'Episode_Termination/physics_diverged')
    print(json.dumps({tag:{key:value for key,value in scalars[tag].items() if key!='events'}
                      for tag in selected if tag in scalars}))


if __name__ == '__main__':
    main()
