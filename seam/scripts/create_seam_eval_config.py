"""Create a config for a NEW corrected SEAM checkpoint stored in Modal volumes."""

import argparse
import copy
from pathlib import Path, PurePosixPath

import yaml


def checkpoint_config(template, run_dir, steps=500, seed=42):
    if PurePosixPath(run_dir).name != run_dir or not run_dir.startswith('seam-full-seed'):
        raise ValueError('Expected a single seam-full-seed... run directory')
    config = copy.deepcopy(template)
    config['models'] = {name: value for name, value in config['models'].items() if name == 'base'}
    config['models']['seam'] = {
        'source': {'kind': 'volume', 'volume': 'seam-full-results',
                   'path': f'{run_dir}/defense/checkpoint-{steps}'},
        'defense_data': [{'kind': 'volume', 'volume': 'seam-full-data',
                          'path': f'{run_dir}/defense.json'}],
    }
    config['protocol']['seed'] = seed
    config['data']['benign']['exclude_seed'] = seed
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    config = checkpoint_config(yaml.safe_load(Path(args.template).read_text(encoding='utf-8')),
                               args.run_dir, args.steps, args.seed)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('# Use after the corresponding training status is complete.\n' +
                    yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    print(f'Eval config: {path.resolve()}')
