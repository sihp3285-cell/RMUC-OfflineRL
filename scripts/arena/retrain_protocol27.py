"""Audit a completed collection, train, then evaluate frozen protocol-v1 student.

Run from repository root with the dl_env interpreter. Refuses incomplete data,
missing action classes, schema mismatch and nonfinite samples. All subprocess
failures stop the pipeline. Existing outputs are never silently overwritten.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from rm_rl.waypoint import protocol27 as p
from rm_rl.waypoint.action_space import load,validate

DATA=ROOT/'data/radar27_protocol_v1'
FILTERED=ROOT/'data/radar27_protocol_v1_four_points'
OUT=ROOT/'rm_runs/radar27_protocol_v1'
OLD=ROOT/'rm_runs/radar27_cls_v1/best.pt'
OLD_SHA='4e9fbb86a0bd4b33a9b60b3ba48dac1ed8e1e2b144d31e2005c5cf796889d9d9'

def write(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));tmp.replace(path)

def audit():
    manifest=json.loads((DATA/'manifest.json').read_text())
    for name,digest in manifest['sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'collection dependency changed: {name}')
    progress=json.loads((DATA/'progress.json').read_text())
    if progress['status']!='complete' or progress['completed']!=40:raise ValueError('collection incomplete')
    results=json.loads((DATA/'results.json').read_text())
    if len(results)!=40 or any(r['teacher_policy_errors'] for r in results):raise ValueError('teacher errors or incomplete results')
    files=sorted(DATA.glob('*/match-*.npz'))
    if len(files)!=40:raise ValueError('wrong shard count')
    spec=load(ROOT/'configs/radar/conservative_four_points.json');anchors=validate(spec)
    val=set(sorted({f.stem for f in files})[::5])
    groups={k:dict(samples=0,excluded_samples=0,classes=np.zeros(4,dtype=int),max_error_m=0.,max_excluded_error_m=0.,valid_sum=np.zeros(len(p.FIELDS)),files=[]) for k in ('train','val')}
    if FILTERED.exists():raise ValueError('filtered data already exists; review before overwriting')
    for side in ('red','blue'):(FILTERED/side).mkdir(parents=True,exist_ok=True)
    expected=json.loads(json.dumps(p.contract()))
    for f in files:
        with np.load(f) as d:
            if str(d['schema'].item())!=p.SCHEMA or json.loads(str(d['observation_contract'].item()))!=expected:raise ValueError(f'contract mismatch: {f}')
            x=d['features'];g=d['goals'];t=d['time']
            if x.shape!=(len(g),p.FEATURE_DIM) or not np.isfinite(x).all() or not np.isfinite(g).all() or not np.all(np.diff(t)>0):raise ValueError(f'invalid samples: {f}')
            if not np.isin(x[:,1::3],[0,1]).all():raise ValueError('invalid masks')
            dist=np.linalg.norm(g[:,None,:]-anchors[None,:,:],axis=2);best=dist.min(axis=1)
            keep=best<=spec['max_assignment_error_m']
            if not keep.any():raise ValueError(f'no in-vocabulary samples: {f}')
            filtered={key:(d[key][keep] if key in ('features','goals','labels','ego','time') else d[key]) for key in d.files}
            np.savez_compressed(FILTERED/f.parent.name/f.name,**filtered)
            x=x[keep];labels=dist[keep].argmin(axis=1)
            entry=groups['val' if f.stem in val else 'train']
            entry['samples']+=len(x);entry['classes']+=np.bincount(labels,minlength=4)
            entry['excluded_samples']+=int((~keep).sum())
            if (~keep).any():entry['max_excluded_error_m']=max(entry['max_excluded_error_m'],float(best[~keep].max()))
            entry['valid_sum']+=x[:,1::3].sum(axis=0);entry['max_error_m']=max(entry['max_error_m'],float(best[keep].max()))
            entry['files'].append(str(f.relative_to(ROOT)))
    for e in groups.values():
        e['classes']=e['classes'].tolist()
        e['valid_fraction_by_field']={f.name:float(v/e['samples']) for f,v in zip(p.FIELDS,e.pop('valid_sum'))}
    report=dict(schema=p.SCHEMA,groups=groups,action_names=spec['names'],
                exclusion_policy='Exclude goals > 0.05 m from every fixed point; stop if exclusions exceed 0.5% in either split. Original data is retained.',
                data_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    write(OUT/'audit.json',report)
    for e in groups.values():
        if min(e['classes'])==0:raise ValueError('missing target class; inspect audit.json and collect additional scenarios')
        if e['excluded_samples']/(e['samples']+e['excluded_samples'])>.005:raise ValueError('too many teacher goals outside four points')
        if e['max_error_m']>spec['max_assignment_error_m']:raise ValueError('teacher goal outside declared four points')
    return report

def run(stage,args):
    write(OUT/'pipeline_status.json',dict(stage=stage,status='running',updated=time.time(),command=args))
    with (OUT/(stage+'.log')).open('x') as log:
        subprocess.run([sys.executable,'-u',*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'pipeline_status.json').exists():raise ValueError('pipeline already exists; review before restarting')
    try:
        if hashlib.sha256(OLD.read_bytes()).hexdigest()!=OLD_SHA:raise ValueError('old baseline checksum changed')
        if '--wait-for-collection' in sys.argv:
            write(OUT/'pipeline_status.json',dict(stage='collection',status='waiting',updated=time.time()))
            deadline=time.monotonic()+7200
            while True:
                progress=json.loads((DATA/'progress.json').read_text())
                if progress['status']=='complete':break
                if progress['status']!='running' or time.monotonic()>deadline:raise ValueError('collection failed or timed out')
                time.sleep(5)
        audit()
        run('train',['-m','rm_rl.waypoint.train_cls','--data',str(FILTERED/'red'),str(FILTERED/'blue'),
                     '--actions',str(ROOT/'configs/radar/conservative_four_points.json'),'--out',str(OUT/'model')])
        # Development seeds never overlap training or final tests.
        common=['-m','rm_rl.waypoint.radar_matrix','--workers','12','--duration','420']
        model=['--ckpt',str(OUT/'model/best.pt'),'--observation-schema',p.SCHEMA]
        run('development',common+model+['--matches','2','--seed-start','14100','--modes','radar','radar_hysteresis','--out',str(OUT/'development')])
        rows=json.loads((OUT/'development/results.json').read_text())
        if any(r['prediction_errors'] or r['candidate_policy_errors'] for r in rows):raise ValueError('development policy errors')
        run('evaluation',common+model+['--matches','5','--seed-start','14200','--out',str(OUT/'evaluation')])
        run('old_reference',common+['--ckpt',str(OLD),'--observation-schema','radar27-ground-v1',
                                   '--matches','5','--seed-start','14200','--modes','radar_hysteresis','--out',str(OUT/'old_reference')])
        if hashlib.sha256(OLD.read_bytes()).hexdigest()!=OLD_SHA:raise ValueError('old baseline checksum changed')
        write(OUT/'pipeline_status.json',dict(stage='complete',status='complete',updated=time.time(),
            note='40 new-system evaluations plus 10 old-contract references. Old/new have different input assumptions; not an equal-observation architecture comparison.'))
    except BaseException as e:
        write(OUT/'pipeline_status.json',dict(stage='stopped',status='failed',error=str(e),updated=time.time()))
        raise

if __name__=='__main__':main()
