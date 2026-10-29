"""Resolve sentry goal before ALL goal-dependent logic in supported Arena executors.

Only the loaded in-memory function is adapted; external strategy source is untouched.
Fail closed on unrecognized source instead of silently reverting to late substitution.
"""
import ast
import inspect
import textwrap

VERSION = 'goal-before-posture-v2'


def install_goal_resolver(module, resolver):
    hook = '_radar_resolve_final_goal'
    if hasattr(module, hook):
        raise ValueError('executor already adapted')
    original = module.sentry
    if getattr(module, 'ROLES', {}).get('sentry', original) is not original:
        raise ValueError('unsupported sentry dispatch table')
    fn = ast.parse(textwrap.dedent(inspect.getsource(module.sentry))).body[0]
    expected = ast.parse("moving = distance(robot, goal) > .6").body[0]
    matches = [i for i, stmt in enumerate(fn.body) if ast.dump(stmt) == ast.dump(expected)]
    if len(matches) != 1 or [a.arg for a in fn.args.args] != ['robot', 'ctx']:
        raise ValueError('unsupported sentry executor: no unique goal-to-moving boundary')
    i = matches[0]
    if any(isinstance(n, ast.Name) and n.id == 'goal' and isinstance(n.ctx, ast.Store)
           for stmt in fn.body[i:] for n in ast.walk(stmt)):
        raise ValueError('unsupported sentry executor: goal overwritten after boundary')
    fn.body.insert(i, ast.parse(f'goal = {hook}(robot, ctx, goal)').body[0])
    tree = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    setattr(module, hook, resolver)
    exec(compile(tree, '<radar-goal-before-posture-v2>', 'exec'), module.__dict__)
    if hasattr(module, 'ROLES'):
        module.ROLES['sentry'] = module.sentry
