"""Isolated source-reviewed safe grammar and construction checks."""
from __future__ import annotations
import ast, hashlib, itertools, json, re
from pathlib import Path
import numpy as np
import sympy as sp
from .symbols import REGISTRY, variable_symbols
FUNCTIONS = {name: getattr(np, name) for name in REGISTRY['functions']}
VARIABLES = set(REGISTRY['variables'])
ROLES = ['THEORY_REEXPRESSION','MECHANISM_EXTENSION','RIVAL_CONTROL']
STR = {'type':'string'}
from .storage import save, read
def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _canonical_mathematics(proposal, protocol):
    """Joint alpha-equivalence of centred expression, domains and conditions.

    The same parameter permutation is applied to every part of the contract.
    This is an exact symbolic identity check, not numerical curve matching.
    The bounded permutation audit changes no admission/parameter-count rule:
    large forms retain a conservative declaration-order key.
    """
    params = proposal['parameters']
    names = [item['name'] for item in params]
    symbols = variable_symbols()
    symbols.update({item['name']: sp.Symbol(item['name'],
        **({'positive': True} if item['domain'] == 'POSITIVE' else {'real': True}))
        for item in params})
    expression = symbolic(proposal['expression'], symbols)
    reference = {symbols[name]: value for name, value in protocol['reference'].items()}
    centered = sp.simplify(expression - expression.subs(reference, simultaneous=True))
    conditions = [(item['relation'], sp.simplify(symbolic(item['expression'], symbols)))
                  for item in proposal['conditions']]
    permutations = itertools.permutations(range(len(params))) if len(params) <= 6 else [tuple(range(len(params)))]
    best = None
    for order in permutations:
        renaming = {symbols[names[index]]: sp.Symbol('p'+str(position),
            **({'positive': True} if params[index]['domain'] == 'POSITIVE' else {'real': True}))
            for position, index in enumerate(order)}
        descriptor = [proposal['hypothesis_id'],
            sp.srepr(centered.xreplace(renaming)),
            [params[index]['domain'] for index in order],
            sorted((relation, sp.srepr(expr.xreplace(renaming))) for relation, expr in conditions)]
        serialized = json.dumps(descriptor, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
        # Parameter order is not a tiebreaker for mathematical identity. A
        # symmetric tie can be resolved by name for stable in-run numerics.
        tie = tuple(names[index] for index in order)
        if best is None or (serialized, tie) < (best[0], best[1]):
            best = (serialized, tie, order)
    return {'key': sha(('MATH_ALPHA_V12:'+best[0]).encode()),
            'parameter_order': [names[index] for index in best[2]],
            'centered_expression': str(centered),
            'scope': 'FULL_JOINT_ALPHA_EQUIVALENCE' if len(params) <= 6 else 'CONSERVATIVE_DECLARATION_ORDER_ABOVE_SIX_PARAMETERS'}


def canonical_mathematical_key(proposal, protocol):
    return _canonical_mathematics(proposal, protocol)['key']


def canonical_parameter_order(proposal, protocol):
    return _canonical_mathematics(proposal, protocol)['parameter_order']


def normalized_fit_key(row, protocol):
    """Numerical identity after exact gauge reduction, with fit coordinates."""
    descriptor = {'version':'NORMALIZED_FIT_V12',
        'mathematics':canonical_mathematical_key(row['proposal'], protocol),
        'reference':protocol['reference'], 'domains':protocol.get('domains', {})}
    return sha(json.dumps(descriptor, sort_keys=True, ensure_ascii=False,
                          allow_nan=False, separators=(',', ':')).encode())


def obj(fields):
    return {'type': 'object', 'additionalProperties': False,
            'properties': fields, 'required': list(fields)}


def arr(item):
    # No module-count quota or minimum is encoded in any array schema.
    return {'type': 'array', 'items': item}


def enum(values):
    return {'type': 'string', 'enum': values}


def schemas():
    evidence_link = obj({'evidence_id': STR, 'supports': STR, 'does_not_support': STR,
                         'transfer_assumptions': STR})
    bridge = obj({'hypothesis_id': enum(['H1', 'H2', 'H3', 'H4']),
        'evidence_links': arr(evidence_link), 'process_arguments': arr(STR),
        'distinct_alternatives_to_test': arr(STR), 'evidence_gaps': arr(STR),
        'excluded_misapplications': arr(STR)})
    source = obj({'evidence_id': STR, 'supports': STR, 'source_to_module_mapping': STR})
    parameter = obj({'name': STR, 'domain': enum(['POSITIVE', 'REAL']), 'meaning': STR,
                     'dimension': enum(['dimensionless'])})
    step = obj({'operation': enum(['SOURCE_RELATION', 'ASSUMPTION', 'EXACT_TRANSFORM', 'APPROXIMATION', 'PROPOSED_CLOSURE']),
                'statement': STR, 'expression_after_step': STR})
    condition = obj({'expression': STR, 'relation': enum(['GT_ZERO', 'GE_ZERO']), 'reason': STR})
    proposal = obj({'proposal_id': STR, 'hypothesis_id': enum(['H1', 'H2', 'H3', 'H4']),
        'source_symbols': arr(obj({'name': STR, 'meaning': STR, 'unit': STR, 'mapping_note': STR})),
        'symbol_definitions': arr(obj({'name': STR, 'expression': STR, 'meaning': STR})),
        'scientific_role': enum(ROLES), 'family': STR, 'expression': STR,
        'construction_basis': enum(['SOURCE_LOG_TRANSFORM', 'REDUCED_PROCESS_BALANCE',
                                   'LOCAL_APPROXIMATION', 'PHENOMENOLOGICAL_CLOSURE', 'RIVAL_CONTROL']),
        'governing_relation': STR, 'rate_expression_before_log': STR,
        'parameters': arr(parameter), 'source_links': arr(source), 'evidence_gap': STR,
        'process_argument': STR, 'construction_steps': arr(step), 'added_assumptions': arr(STR),
        'dimension_argument': STR, 'conditions': arr(condition),
        'derivative_axis': enum(['tau', 'c', 'h']), 'claimed_derivative': STR,
        'trend': enum(['INCREASING', 'DECREASING', 'INTERIOR_MAXIMUM', 'INTERIOR_MINIMUM', 'CONSTANT', 'UNCONSTRAINED']),
        'trend_scope': STR, 'limit_argument': STR, 'discriminating_prediction': STR,
        'falsification': STR, 'novelty_relative_to_alternatives': STR})
    proposals = obj({'proposals': arr(proposal), 'coverage_not_claimed': arr(STR),
                     'why_stop_adding_forms': STR})
    verdict = obj({'proposal_id': STR, 'verdict': enum(['ELIGIBLE', 'ELIGIBLE_WITH_CAVEATS', 'QUARANTINE']),
        'source_fidelity': STR, 'process_to_math_judgment': STR,
        'critical_issues': arr(STR), 'caveats': arr(STR)})
    review = obj({'verdicts': arr(verdict), 'missing_distinct_alternatives': arr(STR),
                  'scope_assessment': STR})
    return obj({'bridges': arr(bridge), 'overall_evidence_limits': STR}), proposals, review


def safe_tree(expression, names):
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError('Empty expression')
    tree = ast.parse(expression, mode='eval')
    if sum(1 for _ in ast.walk(tree)) > 240:
        raise ValueError('AST safety limit')
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult,
               ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Call, ast.Name, ast.Load, ast.Constant)
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError('Unsafe grammar: ' + type(node).__name__)
        if isinstance(node, ast.Name) and node.id not in set(names) | VARIABLES | set(FUNCTIONS):
            raise ValueError('Undeclared identifier: ' + node.id)
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name)
                or node.func.id not in FUNCTIONS or len(node.args) != 1 or node.keywords):
            raise ValueError('Unsafe function call')
        if isinstance(node, ast.Constant) and (isinstance(node.value, bool)
                or not isinstance(node.value, (int, float)) or not np.isfinite(node.value)):
            raise ValueError('Unsafe constant')
    return tree


def symbolic(expression, symbols):
    tree = safe_tree(expression, symbols)
    def convert(n):
        if isinstance(n, ast.Expression): return convert(n.body)
        if isinstance(n, ast.Constant): return sp.Rational(str(n.value))
        if isinstance(n, ast.Name): return symbols[n.id]
        if isinstance(n, ast.UnaryOp): return -convert(n.operand) if isinstance(n.op, ast.USub) else convert(n.operand)
        if isinstance(n, ast.BinOp):
            a, b = convert(n.left), convert(n.right)
            return {ast.Add: lambda: a+b, ast.Sub: lambda: a-b, ast.Mult: lambda: a*b,
                    ast.Div: lambda: a/b, ast.Pow: lambda: a**b}[type(n.op)]()
        if isinstance(n, ast.Call):
            a = convert(n.args[0])
            return {'log': sp.log, 'log1p': lambda x: sp.log(1+x), 'exp': sp.exp,
                    'sqrt': sp.sqrt, 'tanh': sp.tanh}[n.func.id](a)
        raise ValueError('Unsafe symbolic node')
    return convert(tree)


def evaluate(expression, values):
    safe_tree(expression, values)
    with np.errstate(all='ignore'):
        return np.asarray(eval(compile(expression, '<safe-module>', 'eval'),
                               {'__builtins__': {}}, {**FUNCTIONS, **values}), dtype=float)


def verify_one(p, protocol, evidence_ids):
    from .symbol_compiler import compile_proposal
    try:
        compiled, receipt = compile_proposal(p)
    except (ValueError, SyntaxError) as exc:
        return {'proposal_id': p['proposal_id'], 'errors': [str(exc)], 'warnings': [], 'status': 'QUARANTINE'}
    result = _verify_one(compiled, protocol, evidence_ids)
    result['symbol_compilation'] = receipt
    result['compiled_proposal'] = compiled
    return result


def _verify_one(p, protocol, evidence_ids):
    errors, warnings = [], []
    result = {'proposal_id': p.get('proposal_id'), 'errors': errors, 'warnings': warnings,
              'global_physical_proof': False, 'source_page_verified': False}
    try:
        hid = p['hypothesis_id']
        names = [x['name'] for x in p['parameters']]
        if len(set(names)) != len(names) or any(not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', n)
                or n in REGISTRY['parameter_contract']['reserved_names'] for n in names):
            raise ValueError('Invalid or duplicate parameter names')
        if any(x['domain'] not in ['POSITIVE', 'REAL'] or x['dimension'] != 'dimensionless' for x in p['parameters']):
            raise ValueError('Undeclared parameter domain/dimension')
        tree = safe_tree(p['expression'], names)
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        if not set(names) <= used:
            raise ValueError('Unused parameters')
        variables = used & VARIABLES
        valid_scope = {'H1': variables == {'tau'}, 'H2': variables == {'c'},
                       'H3': variables == {'h'}, 'H4': 'h' in variables}
        if not valid_scope.get(hid):
            # A constant no-effect control is legitimate, not a missing-variable error.
            if not (p['scientific_role'] == 'RIVAL_CONTROL' and not variables):
                raise ValueError('Hypothesis/variable scope mismatch')
        if p['derivative_axis'] != ('tau' if hid == 'H1' else 'c' if hid == 'H2' else 'h'):
            raise ValueError('Wrong derivative axis')
        for link in p['source_links']:
            if link['evidence_id'] not in evidence_ids:
                raise ValueError('Unknown evidence ID')
            if not link['supports'].strip() or not link['source_to_module_mapping'].strip():
                raise ValueError('Empty source-to-module explanation')
        if p['scientific_role'] == 'THEORY_REEXPRESSION' and not p['source_links']:
            raise ValueError('Theory reexpression without source')
        if not p['source_links'] and not p['evidence_gap'].strip():
            raise ValueError('Missing evidence and explicit evidence gap')
        for field in ['family', 'process_argument', 'dimension_argument', 'trend_scope',
                      'limit_argument', 'discriminating_prediction', 'falsification',
                      'novelty_relative_to_alternatives']:
            if not p[field].strip(): raise ValueError('Missing construction field: '+field)
        symbols = variable_symbols()
        for item in p['parameters']:
            symbols[item['name']] = sp.Symbol(item['name'], **({'positive': True} if item['domain'] == 'POSITIVE' else {'real': True}))
        expression = symbolic(p['expression'], symbols)
        result['expression_operators'] = sum(isinstance(n,(ast.BinOp,ast.UnaryOp,ast.Call)) for n in ast.walk(tree))
        result['parameter_count'] = len(names)
        basis=p['construction_basis']
        result['construction_basis']=basis
        rate_text=p['rate_expression_before_log'].strip()
        if basis in ['SOURCE_LOG_TRANSFORM','REDUCED_PROCESS_BALANCE'] and not rate_text:
            raise ValueError('Process/source-derived module lacks its pre-log rate relation')
        if not p['governing_relation'].strip():
            raise ValueError('No governing relation or explicit phenomenological closure statement')
        rate=None
        if rate_text:
            rate=symbolic(rate_text,symbols)
            residuals={}
            for v in sorted(variables):
                rr=sp.simplify(sp.diff(expression,symbols[v])-sp.diff(rate,symbols[v])/rate)
                residuals[v]=str(rr)
                if rr!=0:
                    if rr.equals(0) is False: errors.append('Pre-log rate and log-module transformation disagree for '+v)
                    else: warnings.append('Pre-log transformation unresolved for '+v)
            result['pre_log_transform_derivative_residuals']=residuals
            result['pre_log_transform_scope']='Equality up to additive constant on a connected positive-rate domain, provided every variable residual is zero. Physical assumptions are separately reviewed.'
        else:
            result['pre_log_transform_scope']='No process-derived pre-log identity claimed; labelled approximation, phenomenological closure or null rival.'
        if not p['construction_steps']:
            raise ValueError('No construction steps')
        for index, step in enumerate(p['construction_steps']):
            try:
                safe_tree(step['expression_after_step'], names)
            except (ValueError, SyntaxError) as exc:
                raise ValueError(f'Construction step {index}: {exc}') from exc
        last = symbolic(p['construction_steps'][-1]['expression_after_step'], symbols)
        residual = sp.simplify(last-expression)
        result['last_construction_step_residual'] = str(residual)
        if residual != 0:
            errors.append('Final construction step disagrees with proposed expression')
        derivative = sp.diff(expression, symbols[p['derivative_axis']])
        claimed = symbolic(p['claimed_derivative'], symbols)
        dr = sp.simplify(derivative-claimed)
        result['computed_derivative'] = str(derivative)
        result['derivative_residual'] = str(dr)
        if dr != 0:
            if dr.equals(0) is False: errors.append('Claimed derivative is mathematically false')
            else: warnings.append('Derivative identity unresolved symbolically')
        result['dimensions'] = 'DIMENSIONLESS_BY_DECLARED_COORDINATES_AND_PARAMETERS_NOT_SOURCE_DIMENSION_PROOF'
        # Numeric probes are counterexample checks, never proofs or fitted values.
        axes = {k: np.linspace(*protocol['domains'][k], 7) for k in VARIABLES}
        grid = np.meshgrid(*(axes[k] for k in ['tau','c','h']), indexing='ij')
        base = {k: v.reshape(-1) for k, v in zip(['tau','c','h'], grid)}
        probe_reports = []
        for j in range(7):
            parameters = {item['name']: ([-2., -.5, 0., .5, 1., 2., 4.][j]
                if item['domain'] == 'REAL' else [.15, .4, .7, 1., 1.5, 3., 6.][(j+i)%7])
                for i, item in enumerate(p['parameters'])}
            values = {**base, **parameters}
            admissible = np.ones(len(base['tau']), dtype=bool)
            for cond in p['conditions']:
                z = evaluate(cond['expression'], values)
                admissible &= np.isfinite(z) & (z > 0 if cond['relation'] == 'GT_ZERO' else z >= 0)
            n = int(admissible.sum())
            if not n:
                probe_reports.append({'probe': j, 'admissible_grid_points': 0})
                continue
            y = np.broadcast_to(evaluate(p['expression'], values), admissible.shape)
            refvalues = {**protocol['reference'], **parameters}
            refok = all(bool(np.all(np.isfinite(z := evaluate(cond['expression'], refvalues))))
                         and bool(np.all(z > 0 if cond['relation'] == 'GT_ZERO' else z >= 0)) for cond in p['conditions'])
            yr = evaluate(p['expression'], refvalues)
            finite = bool(np.isfinite(y[admissible]).all() and np.isfinite(yr).all())
            positive_rate=None
            if rate_text:
                rv=np.broadcast_to(evaluate(rate_text,values),admissible.shape)
                positive_rate=bool(np.isfinite(rv[admissible]).all() and np.all(rv[admissible]>0))
                if not positive_rate: errors.append('Pre-log rate is not finite and positive at an admissible probe')
            with np.errstate(all='ignore'):
                d = np.broadcast_to(np.asarray(sp.lambdify(list(symbols.values()), derivative, 'numpy')(
                    *(values[k] for k in symbols)), dtype=float), admissible.shape)
            finite_d = bool(np.isfinite(d[admissible]).all())
            trend = p['trend']
            trend_ok = None
            finite_ds = d[admissible & np.isfinite(d)]
            if trend == 'INCREASING': trend_ok = bool(np.all(finite_ds >= -1e-8))
            if trend == 'DECREASING': trend_ok = bool(np.all(finite_ds <= 1e-8))
            if trend == 'CONSTANT': trend_ok = bool(np.all(np.abs(finite_ds) < 1e-8))
            probe_reports.append({'probe': j, 'parameter_values': parameters,
                'admissible_grid_points': n, 'finite_response_and_reference': finite,
                'reference_satisfies_conditions': refok,
                'reference_point': protocol['reference'],
                'pre_log_rate_finite_positive':positive_rate,
                'finite_derivative_on_grid': finite_d, 'claimed_monotone_trend_counterexample': trend_ok is False})
            if not finite: errors.append('Nonfinite response/reference at an admissible probe')
            if not refok: errors.append('REFERENCE_CONDITION_VIOLATION: global feasibility conditions exclude the reference point at a declared parameter probe; local shape conditions belong in trend_scope')
            if not finite_d: warnings.append('Derivative singularity on probe grid; finite response does not imply smoothness')
            if trend_ok is False: errors.append('Counterexample to stated trend within declared parameter conditions')
        if not any(x['admissible_grid_points'] for x in probe_reports):
            errors.append('No admissible probe; cannot establish nonempty construction domain')
        result['numerical_probes'] = probe_reports
        if p['trend'] in ['INTERIOR_MAXIMUM', 'INTERIOR_MINIMUM']:
            warnings.append('Interior-extremum existence/location requires candidate-specific analysis; trend label is not a proof')
        if p['conditions']:
            warnings.append('Additional parameter/environment inequalities must be enforced in later fitting')
        if not variables:
            warnings.append('No-effect control becomes zero after centering; retain as explicit module omission, not fitted coefficient')
        # Canonical equality under parameter renaming + centering. We do not
        # merge scale-reparameterizations or numerically similar curves.
        canonical = _canonical_mathematics(p, protocol)
        result['canonical_centered_expression'] = canonical['centered_expression']
        centered=sp.simplify(expression-expression.subs(
            {symbols[k]:v for k,v in protocol['reference'].items()}, simultaneous=True))
        lost=[item['name'] for item in p['parameters']
              if sp.simplify(sp.diff(centered,symbols[item['name']])) == 0]
        result['parameters_removed_by_centering']=lost
        if lost:
            errors.append('UNIDENTIFIABLE_AFTER_CENTERING: '+','.join(lost))
        result['dedup_key'] = canonical['key']
        result['mathematical_identity_scope'] = canonical['scope']
        if len(names) > 6:
            warnings.append('Alpha-equivalence audit uses declaration order above six parameters; distinct keys need not imply distinct mathematics.')
    except Exception as exc:
        errors.append(type(exc).__name__ + ': ' + str(exc))
    result['errors'] = list(dict.fromkeys(errors))
    result['warnings'] = list(dict.fromkeys(warnings))
    result['status'] = 'QUARANTINE' if errors else 'EXECUTABLE_WITH_CHECKED_CONSTRUCTION'
    return result


def validate_bridges(bridges, evidence_ids):
    if sorted(x['hypothesis_id'] for x in bridges['bridges']) != ['H1','H2','H3','H4']:
        raise ValueError('Four unique scientific-question bridges required')
    for b in bridges['bridges']:
        if not {x['evidence_id'] for x in b['evidence_links']} <= evidence_ids:
            raise ValueError('Unknown bridge evidence ID')
