"""Local radar-input collection and paired closed-loop experiments.

Examples (use the dl_env Python):
  -m rm_rl.waypoint.radar_sim collect --matches 8 --out data/radar27_pilot
  -m rm_rl.waypoint.train_cls --data data/radar27_pilot --out rm_runs/radar27_cls
  -m rm_rl.waypoint.radar_sim evaluate --ckpt rm_runs/radar27_cls_v1/best.pt --out rm_runs/radar27_eval

This is a semantic observation simulator, not a radio/wire parser. The full
snapshot is used only by the sensor boundary and evaluator, never by student
or rule prediction. Student, rule and baseline groups share the executor;
the optional teacher comparison uses its native strategy.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from rm_rl.waypoint.radar import (SCHEMA, ANCHORS, RadarConfig,
                                  canonical_xy, value, FIELD_INDEX)
from rm_rl.waypoint.teachers import StrategyRunner, TracedSentryTeacher
from rm_rl.waypoint.model import WaypointClassifier
from rm_rl.waypoint.goal_filter import GoalFilter
from rm_rl.waypoint.executor import install_goal_resolver, VERSION as EXECUTOR_VERSION
from rm_rl.waypoint.action_space import validate as validate_actions
from rm_rl.waypoint.observation import SCHEMAS, description, adapter as make_adapter, reader, field_index

APP = '/home/delphine/rm/arena-runtime/squashfs-root/resources/runtime/app'
TEACHER = '/home/delphine/.config/Arena-1/.arena_data/strategy-library/versions/17978d64f27265c1a3acc7c074f0802bb4b3ee349598428c87136c0e5255d72f/strategies/gpt61_2027_adaptive_2772/strategy.py'
BASELINE = '/home/delphine/rm/arena_strategy_backup/strategy3_baseline.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup(args, seed):
    sys.path.insert(0, args.app)
    sdk = str(Path(args.app) / 'rmuc_arena')
    from rmuc_arena.engine import Match
    from rmuc_arena.models import mirrored_lineups
    lineups = mirrored_lineups({'1': 2711, '3': 2731, '4': 2732, '6': 2761, '7': 2771}, '2027')
    m = Match(seed=seed, duration=args.duration, map_name='rmuc2027', lineups=lineups)
    base = StrategyRunner(args.baseline, sdk)
    opponent = StrategyRunner(args.opponent, sdk)
    return m, base, opponent, sdk


def config(args):
    return RadarConfig(latency_s=args.latency, loss=args.loss, position_noise_m=args.noise,
                       sentry_relay=not args.no_sentry_relay, ally_relay=not args.no_ally_relay,
                       enemy_feed=not args.no_enemy_feed)


def commands(m, side, candidate, state, base, bs, opponent, os):
    other = 'blue' if side == 'red' else 'red'
    obs = m.observation(side, [7])
    acts, state = candidate.act(obs, state)
    allies, bs = base.act(m.observation(side, [1, 3, 4, 6]), bs)
    foes, os = opponent.act(m.observation(other, [1, 3, 4, 6, 7]), os)
    own = {str(k): v for k, v in (allies or {}).items() if str(k) in ('1', '3', '4', '6')}
    own.update({str(k): v for k, v in (acts or {}).items() if str(k) == '7'})
    return {side: own, other: foes}, state, bs, os


class Predictor:
    def __init__(self, path, expected):
        state = torch.load(path, map_location='cpu', weights_only=False)
        if expected is None: expected = state.get('observation_schema')
        if expected not in SCHEMAS:
            raise ValueError('unsupported radar checkpoint schema')
        if state.get('observation_schema', 'arena307') != expected:
            raise ValueError(f'checkpoint schema must be {expected}')
        if expected in SCHEMAS and (state['obs_dim'] != description(expected)[0] or state['observation_contract'] != description(expected)[1]):
            raise ValueError('radar checkpoint contract mismatch')
        self.schema = expected
        self.model = WaypointClassifier(state['obs_dim'], state['n_classes'], state['hidden'], state['depth'])
        self.model.load_state_dict(state['state_dict']); self.model.eval()
        self.mean, self.std = state['mean'], state['std']
        self.anchors = np.asarray(state['anchors'])
        action_spec = state.get('action_contract')
        if action_spec is not None:
            if not np.array_equal(validate_actions(action_spec), self.anchors):
                raise ValueError('checkpoint action contract mismatch')
            self.supply_anchor = self.anchors[action_spec['supply_index']]
        else:
            self.supply_anchor = ANCHORS[3]

    def predict(self, features):
        x = (features - self.mean) / self.std
        if not np.isfinite(x).all():
            raise ValueError('nonfinite model input')
        with torch.no_grad():
            logits = self.model(torch.from_numpy(x.astype(np.float32)).unsqueeze(0)).numpy()[0]
        if not np.isfinite(logits).all():
            raise ValueError('nonfinite model output')
        return self.anchors[int(np.argmax(logits))]


def rule_goal(f, schema=SCHEMA):
    value = reader(schema)
    hp, max_hp, ammo = (value(f, n) for n in ('ally.7.hp', 'ally.7.max_hp', 'ally.7.ammo_proxy'))
    if ammo is not None and ammo < 30 or hp is not None and max_hp and hp / max_hp < .24:
        return ANCHORS[3]
    # Explicit, modest rule baseline using exactly the same observation vector.
    threat = False
    for slot in (1, 3, 4, 7):
        x, y, alive = (value(f, f'enemy.{slot}.{n}') for n in ('x', 'y', 'alive'))
        if x is not None and y is not None and alive and np.hypot(x - 10.8, y - 7.5) < 5:
            threat = True
    if threat or (value(f, 'ally.central.occupied_enemy') or 0) > .5:
        return ANCHORS[0]
    return ANCHORS[1]


class Candidate:
    def __init__(self, args, sdk, mode):
        self.module = StrategyRunner(args.teacher if mode == 'teacher' else args.baseline, sdk).module
        self.policy_errors = 0
        self.policy_error_messages = []
        # The baseline act() catches decide() exceptions internally. Observe
        # them before that catch, preserving its original fallback behavior.
        if hasattr(self.module, 'decide'):
            original_decide = self.module.decide
            def observed_decide(*a, **kw):
                try:
                    return original_decide(*a, **kw)
                except Exception as exc:
                    self.policy_errors += 1
                    if len(self.policy_error_messages) < 5:
                        self.policy_error_messages.append(str(exc))
                    raise
            self.module.decide = observed_decide
        self.original = self.module.drive
        self.module.drive = self.drive
        self.mode, self.side = mode, args.side
        self.schema = getattr(args, "observation_schema", SCHEMA)
        self.value = reader(self.schema)
        self.predictor = Predictor(args.ckpt, getattr(args, "observation_schema", SCHEMA)) if mode in ('radar', 'radar_hysteresis') else None
        self.filter = GoalFilter(args.confirm_s, args.min_hold_s) if mode == 'radar_hysteresis' else None
        self.proposed = None
        self.filter_reason = None
        self.held_decisions = self.urgent_decisions = 0
        self.goal = None
        self.selected = None
        self.features = None
        self.fallbacks = 0
        self.prediction_errors = 0
        self.decisions = 0
        self.executor = getattr(args, 'executor', 'legacy')
        if self.executor == EXECUTOR_VERSION and mode != 'teacher':
            install_goal_resolver(self.module, lambda robot, ctx, goal: self.selected if self.selected is not None else goal)

    def drive(self, robot, goal, obs, memory, tolerance=.35):
        if str(robot.get('slot')) == '7' and robot.get('role') == 'sentry':
            if self.selected is not None and self.executor == 'legacy':
                goal = self.selected
            self.goal = (float(goal[0]), float(goal[1]))
        return self.original(robot, goal, obs, memory, tolerance)

    def act(self, obs, state):
        self.goal = self.selected = self.proposed = None
        self.filter_reason = None
        self.decisions += 1
        if self.mode in ('radar', 'radar_hysteresis', 'rule'):
            if any(self.value(self.features, 'ally.7.' + key) is None for key in ('x', 'y')):
                self.fallbacks += 1
                if self.filter: self.filter.reset()
                self.filter_reason = 'missing_position'
            else:
                try:
                    p = self.predictor.predict(self.features) if self.predictor else rule_goal(self.features, self.schema)
                    self.proposed = tuple(map(float, p))
                    if self.filter:
                        if self.value(self.features, 'ally.7.alive') == 0:
                            self.filter.reset()
                            self.filter_reason = 'dead_reset'
                        else:
                            hp, max_hp, ammo = (self.value(self.features, n) for n in ('ally.7.hp', 'ally.7.max_hp', 'ally.7.ammo_proxy'))
                            urgent = bool(np.allclose(p, self.predictor.supply_anchor) and (ammo is not None and ammo < 30 or hp is not None and max_hp and hp / max_hp < .24))
                            p = self.filter.update(p, float(obs['time']), urgent=urgent)
                            self.filter_reason = self.filter.reason
                            self.held_decisions += int(self.filter_reason == 'held')
                            self.urgent_decisions += int(self.filter_reason == 'urgent')
                    self.selected = canonical_xy(float(p[0]), float(p[1]), self.side)
                except (ValueError, RuntimeError):
                    if self.filter: self.filter.reset()
                    self.filter_reason = 'prediction_error'
                    self.prediction_errors += 1
                    self.fallbacks += 1
        if self.selected is not None and not (0 <= self.selected[0] <= 28 and 0 <= self.selected[1] <= 15):
            raise ValueError('invalid waypoint')
        return self.module.act(obs, state)


def sample(frame, side):
    r = next(r for r in frame['robots'] if r['team'] == side and int(r['slot']) == 7)
    return {k: r.get(k, 0) for k in ('x', 'y', 'hp', 'alive', 'ammo', 'heat', 'max_heat', 'damage', 'kills', 'vx', 'vy')}


def metrics(history):
    alive = overheat = taken = stuck = 0.0
    deaths = 0
    segments = []
    previous_goal = None
    for row in history:
        a, b, dt, goal = row['before'], row['after'], row['dt'], row['goal']
        if a['alive']:
            alive += dt
            taken += max(0., a['hp'] - b['hp'])  # includes lethal transition; still a net-HP proxy
            deaths += int(not b['alive'])
            overheat += dt * (a['max_heat'] > 0 and a['heat'] >= .95 * a['max_heat'])
        if not a['alive']:
            previous_goal = None
            continue
        if goal is not None:
            if previous_goal is None or np.linalg.norm(np.subtract(goal, previous_goal)) > .5:
                segments.append({'hit': False, 'goal': goal, 'start': row['t'], 'reach_s': None})
            previous_goal = goal
            if b['alive'] and np.hypot(b['x'] - goal[0], b['y'] - goal[1]) <= .8:
                seg = segments[-1]
                if not seg['hit']:
                    seg['reach_s'] = row['t'] + dt - seg['start']
                seg['hit'] = True
            if np.hypot(a['x'] - goal[0], a['y'] - goal[1]) > 1 and np.hypot(a['vx'], a['vy']) < .1:
                stuck += dt
        else:
            previous_goal = None
    last = history[-1]['after'] if history else {}
    damage = float(last.get('damage', 0)) - (float(history[0]['before']['damage']) if history else 0)
    # True switch count excludes initial issuance and restart after no-command/death.
    switches = sum(1 for a, b in zip(history, history[1:]) if a['goal'] is not None and b['goal'] is not None
                   and a['after']['alive'] and b['before']['alive']
                   and np.linalg.norm(np.subtract(a['goal'], b['goal'])) > .5)
    return {'damage_dealt': damage, 'damage_taken_hp_proxy': taken, 'deaths': deaths,
            'alive_time_s': alive, 'overheat_s': overheat, 'stuck_cmd_s': stuck,
            'goal_switches': switches, 'goal_segments': len(segments),
            'reached_segments': sum(s['hit'] for s in segments),
            'mean_reach_s': float(np.mean([s['reach_s'] for s in segments if s['hit']])) if any(s['hit'] for s in segments) else None,
            'arrival_rate': sum(s['hit'] for s in segments) / len(segments) if segments else None,
            'damage_dealt_per_alive_min': damage * 60 / alive if alive else None}


def collect_one(args, seed):
    m, base, opponent, sdk = setup(args, seed)
    adapter = make_adapter(getattr(args, "observation_schema", SCHEMA), config(args), args.side, seed)
    teacher = TracedSentryTeacher(args.teacher, sdk)
    state = bs = os = None
    rows, goals, egos, times = [], [], [], []
    try:
        while m.status != 'finished':
            frame = m.snapshot()
            f = adapter.observe(frame)
            cmd, state, bs, os = commands(m, args.side, teacher, state, base, bs, opponent, os)
            g = teacher.last_goal
            if g is not None and 0 <= g[0] <= 28 and 0 <= g[1] <= 15:
                rows.append(f); goals.append(canonical_xy(*g, args.side))
                r = sample(frame, args.side)
                egos.append(canonical_xy(r['x'], r['y'], args.side)); times.append(frame['time'])
            for _ in range(2):
                m.step(cmd)
                if m.status == 'finished': break
        if not rows: raise RuntimeError('no valid teacher targets')
        dest = Path(args.out) / f'match-{seed:04d}.npz'
        goals = np.asarray(goals, dtype=np.float32)
        np.savez_compressed(dest, schema=np.asarray(getattr(args, "observation_schema", SCHEMA)), features=np.asarray(rows), goals=goals,
                            labels=goals / [28, 15], seed=np.asarray(seed), side=np.asarray(args.side),
                            observation_contract=np.asarray(json.dumps(description(getattr(args, "observation_schema", SCHEMA))[1])),
                            teacher_sha256=np.asarray(sha(args.teacher)), ego=np.asarray(egos, dtype=np.float32), time=np.asarray(times),
                            sensor_config=np.asarray(json.dumps(asdict(config(args)))))
        if teacher.policy_errors:
            raise RuntimeError(f'teacher had {teacher.policy_errors} hidden policy errors: {teacher.policy_error_messages}')
        return {'seed': seed, 'side': args.side, 'mode': 'collect', 'teacher_policy_errors': teacher.policy_errors, 'samples': len(rows), 'path': str(dest), 'engine_hash': frame.get('engine_hash'), 'rules_hash': frame.get('rules_hash')}
    finally:
        m.close()


def evaluate_one(args, seed, mode):
    m, base, opponent, sdk = setup(args, seed)
    adapter = make_adapter(getattr(args, "observation_schema", SCHEMA), config(args), args.side, seed)
    candidate = Candidate(args, sdk, mode)
    state = bs = os = None
    history, latencies, valid_fractions, input_ages = [], [], [], []
    try:
        while m.status != 'finished':
            frame = m.snapshot()
            candidate.features = adapter.observe(frame)
            valid_fractions.append(float(candidate.features[1::3].mean()))
            own_i = 3 * field_index(getattr(args, "observation_schema", SCHEMA), "ally.7.x")
            if candidate.features[own_i + 1] > .5:
                input_ages.append(float(candidate.features[own_i + 2]) * 5)
            before = sample(frame, args.side)
            t0 = time.perf_counter()
            cmd, state, bs, os = commands(m, args.side, candidate, state, base, bs, opponent, os)
            latencies.append(1000 * (time.perf_counter() - t0))
            for _ in range(2):
                m.step(cmd)
                if m.status == 'finished': break
            after = m.snapshot()
            history.append({'t': frame['time'], 'dt': after['time'] - frame['time'], 'before': before,
                            'after': sample(after, args.side), 'goal': candidate.goal,
                            'proposed_canonical': candidate.proposed, 'filter_reason': candidate.filter_reason})
        result = {'seed': seed, 'mode': mode, 'side': args.side, 'winner': m.winner,
                  'duration_s': m.time, 'metrics': metrics(history), 'fallbacks': candidate.fallbacks,
                  'held_decisions': candidate.held_decisions, 'urgent_decisions': candidate.urgent_decisions,
                  'executor': candidate.executor, 'prediction_errors': candidate.prediction_errors, 'decisions': candidate.decisions,
                  'candidate_policy_errors': candidate.policy_errors,
                  'candidate_policy_error_messages': candidate.policy_error_messages,
                  'observation_valid_fraction': float(np.mean(valid_fractions)),
                  'sentry_input_age_p99_s': float(np.percentile(input_ages, 99)) if input_ages else None,
                  'sensor_packets_sent': adapter.sent, 'sensor_packets_dropped': adapter.dropped,
                  'all_policies_and_ipc_p99_ms': float(np.percentile(latencies, 99)),
                  'engine_hash': after.get('engine_hash'), 'rules_hash': after.get('rules_hash')}
        with gzip.open(Path(args.out) / f'trace-{mode}-{seed}.json.gz', 'wt') as out:
            json.dump(history, out)
        return result
    finally:
        m.close()


def summarize(results):
    groups = sorted({r['mode'] for r in results})
    reference = 'baseline' if 'baseline' in groups else 'radar'
    base = {r['seed']: r['metrics'] for r in results if r['mode'] == reference}
    summary = {}
    rng = np.random.default_rng(0)
    for group in groups:
        rows = [r for r in results if r['mode'] == group]
        entry = {}
        for key in rows[0]['metrics']:
            vals = [r['metrics'][key] for r in rows if r['metrics'][key] is not None]
            item = {'mean': float(np.mean(vals)) if vals else None}
            diffs = [r['metrics'][key] - base[r['seed']][key] for r in rows if r['seed'] in base
                     and r['metrics'][key] is not None and base[r['seed']][key] is not None]
            if diffs and group != reference:
                item['paired_delta_vs_' + reference] = float(np.mean(diffs))
                item['paired_bootstrap_95_ci'] = np.percentile(rng.choice(diffs, (2000, len(diffs))).mean(axis=1), [2.5, 97.5]).tolist() if len(diffs) >= 2 else None
            entry[key] = item
        summary[group] = entry
    return summary


def run_job(args, seed, mode):
    torch.set_num_threads(1)
    return collect_one(args, seed) if mode == "collect" else evaluate_one(args, seed, mode)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('collect', 'evaluate'))
    p.add_argument('--app', default=APP)
    p.add_argument('--teacher', default=TEACHER)
    p.add_argument('--baseline', default=BASELINE)
    p.add_argument('--opponent', default=APP + '/examples/claude_opus55_xhigh_2027_hold/strategy.py')
    p.add_argument('--ckpt', default='rm_runs/radar27_cls_v1/best.pt')
    p.add_argument('--modes', nargs='+', choices=('baseline', 'rule', 'radar', 'radar_hysteresis', 'teacher'), default=['radar_hysteresis'])
    p.add_argument('--observation-schema', choices=SCHEMAS, default=SCHEMA)
    p.add_argument('--executor', choices=('legacy', EXECUTOR_VERSION), default=EXECUTOR_VERSION)
    p.add_argument('--confirm-s', type=float, default=1.)
    p.add_argument('--min-hold-s', type=float, default=2.)
    p.add_argument('--side', choices=('red', 'blue'), default='red')
    p.add_argument('--matches', type=int, default=4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed-start', type=int, default=8000)
    p.add_argument('--duration', type=int, default=420)
    p.add_argument('--latency', type=float, default=.13)
    p.add_argument('--loss', type=float, default=.01)
    p.add_argument('--noise', type=float, default=0)
    p.add_argument('--no-sentry-relay', action='store_true')
    p.add_argument('--no-ally-relay', action='store_true')
    p.add_argument('--no-enemy-feed', action='store_true')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    if args.matches < 1 or args.duration < 1 or args.workers < 1: p.error('positive matches/duration required')
    config(args)
    GoalFilter(args.confirm_s, args.min_hold_s)
    torch.set_num_threads(1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit('use a fresh output directory to preserve previous experiments')
    manifest = {'args': vars(args), 'sensor_config': asdict(config(args)), 'contract': description(args.observation_schema)[1],
                'code_sha256': {str(path): sha(path) for path in Path(__file__).parent.glob('*.py')},
                'baseline_sha256': sha(args.baseline), 'opponent_sha256': sha(args.opponent),
                'metric_notes': ['damage_taken_hp_proxy includes lethal HP loss but misses simultaneous healing/damage',
                                 'arrival measured per actual drive goal segment; no-command and death end a segment',
                                 'time quantities use actual elapsed engine time; results are simulation evidence only']}
    if args.command == 'collect' or 'teacher' in args.modes: manifest['teacher_sha256'] = sha(args.teacher)
    if args.command == 'evaluate':
        manifest['checkpoint_sha256'] = {m: sha(path) for m, path in [('radar', args.ckpt), ('radar_hysteresis', args.ckpt)] if m in args.modes}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    results = []
    jobs = [(seed, mode) for seed in range(args.seed_start, args.seed_start + args.matches)
            for mode in (['collect'] if args.command == 'collect' else args.modes)]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(run_job, args, seed, mode) for seed, mode in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            results.sort(key=lambda r: (r['seed'], r.get('mode', 'collect')))
            print(json.dumps(result), flush=True)
            (out / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
    if args.command == 'evaluate':
        (out / 'summary.json').write_text(json.dumps(summarize(results), ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
