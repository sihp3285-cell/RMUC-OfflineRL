"""SOP run: ordinary balanced sentry -> protocol-v1 two-point student.

40 teacher matches, grouped validation, strict target audit, 40 training epochs,
8 development matches, then 50 final matches with a frozen protocol-v1 reference.
No unsupported labels are filtered or silently mapped into the action vocabulary.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from rm_rl.waypoint import protocol27 as protocol
from rm_rl.waypoint.action_space import load,validate
from rm_rl.waypoint.radar_sim import BASELINE

DATA=ROOT/'data/radar27_baseline_v1'
OUT=ROOT/'rm_runs/radar27_baseline_v1'
ACTIONS=ROOT/'configs/radar/baseline_two_points.json'
REFERENCE=ROOT/'rm_runs/radar27_protocol_v1/model/best.pt'
PRESERVED=[REFERENCE,ROOT/'rm_runs/radar27_cls_v1/best.pt']

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));tmp.replace(path)

def run(stage,args):
    write(OUT/'pipeline_status.json',dict(stage=stage,status='running',updated=time.time(),command=args))
    with (OUT/(stage+'.log')).open('x') as log:
        subprocess.run([sys.executable,'-u',*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)

def audit():
    manifest=json.loads((DATA/'manifest.json').read_text())
    for name,digest in manifest['sha256'].items():
        if sha(Path(name))!=digest:raise ValueError(f'collection dependency changed: {name}')
    rows=json.loads((DATA/'results.json').read_text())
    progress=json.loads((DATA/'progress.json').read_text())
    if progress['status']!='complete' or len(rows)!=40 or any(r['teacher_policy_errors'] for r in rows):raise ValueError('incomplete or faulty collection')
    files=sorted(DATA.glob('*/match-*.npz'))
    if len(files)!=40:raise ValueError('wrong shard count')
    spec=load(ACTIONS);anchors=validate(spec)
    validation=set(sorted({f.stem for f in files})[::5])
    groups={k:dict(samples=0,classes=np.zeros(len(anchors),dtype=int),max_error_m=0.,valid_sum=np.zeros(len(protocol.FIELDS)),files=[]) for k in ('train','val')}
    expected=json.loads(json.dumps(protocol.contract()))
    for f in files:
        with np.load(f) as d:
            if str(d['schema'].item())!=protocol.SCHEMA or json.loads(str(d['observation_contract'].item()))!=expected:raise ValueError('input contract mismatch')
            if str(d['teacher_sha256'].item())!=sha(Path(BASELINE)):raise ValueError('wrong teacher')
            x,g,t=d['features'],d['goals'],d['time']
            if x.shape!=(len(g),protocol.FEATURE_DIM) or not np.isfinite(x).all() or not np.isfinite(g).all() or not np.all(np.diff(t)>0):raise ValueError('invalid samples')
            if not np.isin(x[:,1::3],[0,1]).all():raise ValueError('invalid masks')
            dist=np.linalg.norm(g[:,None,:]-anchors[None,:,:],axis=2)
            e=groups['val' if f.stem in validation else 'train']
            e['samples']+=len(g);e['classes']+=np.bincount(dist.argmin(axis=1),minlength=len(anchors))
            e['max_error_m']=max(e['max_error_m'],float(dist.min(axis=1).max()))
            e['valid_sum']+=x[:,1::3].sum(axis=0);e['files'].append(str(f.relative_to(ROOT)))
    for e in groups.values():
        e['classes']=e['classes'].tolist()
        e['valid_fraction_by_field']={f.name:float(v/e['samples']) for f,v in zip(protocol.FIELDS,e.pop('valid_sum'))}
    write(OUT/'audit.json',dict(schema=protocol.SCHEMA,action_names=spec['names'],groups=groups,
          data_sha256={str(f.relative_to(ROOT)):sha(f) for f in files},
          limitation='Teacher can use own/ally weakness, unavailable to student. No hidden weakness input added.'))
    for e in groups.values():
        if min(e['classes'])==0:raise ValueError('missing target class; collect additional scenarios')
        if e['max_error_m']>spec['max_assignment_error_m']:raise ValueError('teacher target outside two-point vocabulary')

def verify_errors(directory):
    rows=json.loads((directory/'results.json').read_text())
    if any(r['prediction_errors'] or r['candidate_policy_errors'] for r in rows):raise ValueError(f'policy errors: {directory}')

def main():
    if OUT.exists() or DATA.exists():raise ValueError('experiment already exists; no overwrite')
    OUT.mkdir(parents=True)
    frozen={str(p):sha(p) for p in PRESERVED}
    write(OUT/'experiment.json',dict(teacher=BASELINE,teacher_sha256=sha(Path(BASELINE)),
        actions=load(ACTIONS),input_schema=protocol.SCHEMA,preserved_sha256=frozen,
        collection_seeds=[15000,15019],development_seeds=[15100,15101],test_seeds=[15200,15204],
        workers=12,latency_per_hop_s=.13,loss_per_hop=.01,confirm_s=1,min_hold_s=2,
        final_matches=50,script_sha256=sha(Path(__file__)),
        comparison='Same protocol-v1 sensor/executor/opponent. Current protocol-v1 conservative hysteresis reference, not legacy315 input.'))
    common=['-m','rm_rl.waypoint.radar_matrix','--teacher',BASELINE,'--observation-schema',protocol.SCHEMA,
            '--workers','12','--duration','420','--latency','.13','--loss','.01','--confirm-s','1','--min-hold-s','2']
    try:
        run('collection',common+['--command','collect','--matches','20','--seed-start','15000','--out',str(DATA)])
        audit()
        run('train',['-m','rm_rl.waypoint.train_cls','--data',str(DATA/'red'),str(DATA/'blue'),
                     '--actions',str(ACTIONS),'--out',str(OUT/'model'),'--epochs','40'])
        model=['--ckpt',str(OUT/'model/best.pt')]
        run('development',common+model+['--matches','2','--seed-start','15100','--modes','radar','radar_hysteresis','--out',str(OUT/'development')])
        verify_errors(OUT/'development')
        run('evaluation',common+model+['--matches','5','--seed-start','15200','--out',str(OUT/'evaluation')])
        verify_errors(OUT/'evaluation')
        run('reference',common+['--ckpt',str(REFERENCE),'--matches','5','--seed-start','15200','--modes','radar_hysteresis','--out',str(OUT/'reference')])
        verify_errors(OUT/'reference')
        for name,digest in frozen.items():
            if sha(Path(name))!=digest:raise ValueError('preserved checkpoint changed')
        summary={key:json.loads((OUT/key/'summary.json').read_text()) for key in ('evaluation','reference')}
        write(OUT/'comparison.json',summary)
        write(OUT/'pipeline_status.json',dict(stage='complete',status='complete',updated=time.time(),final_matches=50))
    except BaseException as e:
        write(OUT/'pipeline_status.json',dict(stage='stopped',status='failed',error=str(e),updated=time.time()))
        raise

if __name__=='__main__':main()
