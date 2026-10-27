"""Resumable, paired red/blue radar evaluation; no training or parameter search."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from rm_rl.waypoint import radar_sim as sim
from rm_rl.waypoint.radar import contract

MODES = ('baseline', 'rule', 'radar', 'radar_hysteresis')
SIDES = ('red', 'blue')


def write_json(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    tmp.replace(path)


def aggregate(rows):
    """Bootstrap seeds, keeping the two sides of each seed in the same cluster."""
    means = {}
    for side in (*SIDES, 'both'):
        means[side] = {}
        for mode in MODES:
            group = [r for r in rows if r['mode'] == mode and (side == 'both' or r['side'] == side)]
            means[side][mode] = {'n': len(group)}
            if not group: continue
            for key in group[0]['metrics']:
                values = [r['metrics'][key] for r in group if r['metrics'][key] is not None]
                means[side][mode][key] = float(np.mean(values)) if values else None
            means[side][mode]['win_rate'] = sum(r['winner'] == r['side'] for r in group) / len(group)
    indexed = {(r['seed'], r['side'], r['mode']): r for r in rows}
    paired = {}
    for ref in ('baseline', 'rule', 'radar'):
        paired[ref] = {}
        for side in (*SIDES, 'both'):
            sides = SIDES if side == 'both' else (side,)
            seeds = sorted({r['seed'] for r in rows if all((r['seed'], s, m) in indexed for s in sides for m in (ref, 'radar_hysteresis'))})
            entry = {'complete_seed_pairs': len(seeds)}
            if seeds:
                for key in rows[0]['metrics']:
                    deltas = []
                    for seed in seeds:
                        pairs = [(indexed[seed,s,'radar_hysteresis']['metrics'][key], indexed[seed,s,ref]['metrics'][key]) for s in sides]
                        if all(a is not None and b is not None for a,b in pairs):
                            deltas.append(float(np.mean([a-b for a,b in pairs])))
                    rng = np.random.default_rng(0)
                    entry[key] = {'n_seeds':len(deltas), 'mean_delta':float(np.mean(deltas)) if deltas else None,
                                  'seed_bootstrap_95_ci':np.percentile(rng.choice(deltas,(10000,len(deltas))).mean(axis=1),[2.5,97.5]).tolist() if len(deltas)>1 else None,
                                  'positive_seed_deltas':sum(d>0 for d in deltas), 'negative_seed_deltas':sum(d<0 for d in deltas)}
            paired[ref][side] = entry
    return {'means':means, 'hysteresis_minus_reference':paired,
            'error_totals':{k:sum(r[k] for r in rows) for k in ('prediction_errors','candidate_policy_errors')},
            'notes':['Confidence intervals resample seeds; red and blue remain paired within each seed.',
                     'Damage is the primary screening endpoint; deaths, survival, wins and switching are secondary.',
                     'Baseline uses its original Arena observations; rule and both models share the radar observation boundary.',
                     'Arrival rates depend on the number of goal segments; they are not a standalone success criterion.',
                     'One opponent and provisional radar semantics; these results cannot establish real-world deployment efficacy.']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--matches',type=int,default=30)
    p.add_argument('--seed-start',type=int,default=8400)
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--duration',type=int,default=420)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if min(a.matches,a.workers,a.duration)<1:p.error('positive counts required')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    common=dict(app=sim.APP,teacher=sim.TEACHER,baseline=sim.BASELINE,
                opponent=sim.APP+'/examples/claude_opus55_xhigh_2027_hold/strategy.py',
                ckpt='rm_runs/radar27_cls_v1/best.pt',old_ckpt='rm_runs/waypoint_legal_cls/best.pt',
                duration=a.duration,latency=.13,loss=.01,noise=0.,no_sentry_relay=False,
                no_ally_relay=False,no_enemy_feed=False,confirm_s=1.,min_hold_s=2.)
    fingerprint={'common':common,'seeds':list(range(a.seed_start,a.seed_start+a.matches)), 'modes':MODES,'sides':SIDES,
                 'contract':contract(),'sensor_config':asdict(sim.config(SimpleNamespace(**common))),
                 'sha256':{str(f):sim.sha(f) for f in [*Path(sim.__file__).parent.glob('*.py'),Path(common['ckpt']),Path(common['baseline']),Path(common['opponent']),Path(sim.APP)/'config/rules.json',Path(sim.APP)/'simulation/target/release/rmuc-sim']}}
    fingerprint=json.loads(json.dumps(fingerprint))
    manifest=out/'manifest.json'
    if manifest.exists():
        if not a.resume:p.error('existing run requires --resume')
        if json.loads(manifest.read_text())!=fingerprint:p.error('resume refused: configuration or source hashes changed')
    else:
        if any(out.iterdir()):p.error('output directory is not empty')
        write_json(manifest,fingerprint)
    for side in SIDES:(out/side).mkdir(exist_ok=True)
    rows=[];jobs=[]
    for seed in fingerprint['seeds']:
        for side in SIDES:
            for mode in MODES:
                result=out/side/f'result-{mode}-{seed}.json'
                if result.exists():
                    row=json.loads(result.read_text())
                    if (row['seed'],row['side'],row['mode'])!=(seed,side,mode):raise ValueError('result identity mismatch')
                    if not (out/side/f'trace-{mode}-{seed}.json.gz').exists():raise ValueError('missing saved trace')
                    rows.append(row)
                else:jobs.append((seed,side,mode))
    total=a.matches*len(SIDES)*len(MODES)
    def save(status):
        rows.sort(key=lambda r:(r['seed'],r['side'],r['mode']))
        write_json(out/'results.json',rows)
        write_json(out/'progress.json',dict(status=status,completed=len(rows),total=total,workers=a.workers))
    save('running')
    try:
        with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures={pool.submit(sim.run_job,SimpleNamespace(**common,side=side,out=str(out/side)),seed,mode):(seed,side,mode) for seed,side,mode in jobs}
            for future in as_completed(futures):
                row=future.result();rows.append(row)
                write_json(out/row['side']/f"result-{row['mode']}-{row['seed']}.json",row)
                save('running')
                print(json.dumps(dict(completed=len(rows),total=total,seed=row['seed'],side=row['side'],mode=row['mode'],damage=row['metrics']['damage_dealt'],deaths=row['metrics']['deaths'],errors=row['prediction_errors']+row['candidate_policy_errors'])),flush=True)
        write_json(out/'summary.json',aggregate(rows));save('complete')
    except BaseException:
        save('interrupted_or_failed');raise

if __name__=='__main__':main()
