"""Exact identities only; no fitted values enter reductions."""
import copy, hashlib, itertools, json
import sympy as sp
from .numerics import Module
from . import construction, generation
def math_profile(row):
    proposal = row['proposal']
    symbols = {v: sp.Symbol(v) for v in ('tau', 'c', 'h')}
    symbols.update({p['name']: sp.Symbol(p['name']) for p in proposal['parameters']})
    expr = construction.symbolic(proposal['expression'], symbols)
    variables = {symbols[v] for v in ('tau', 'c', 'h')}
    expanded = sp.expand_mul(expr)
    pieces = [term for term in sp.Add.make_args(expanded) if term.free_symbols & variables]
    canonical = sp.factor_terms(sp.Add(*pieces))
    return {'canonical_expression_before_reference_offset': str(canonical),
            'nodes': sum(1 for _ in sp.preorder_traversal(canonical)),
            'constant_terms_absorbed_by_intercept': len(sp.Add.make_args(expanded)) - len(pieces)}


def variable_part(expr, variables):
    """Remove only additive input-independent terms; the intercept absorbs them."""
    return sp.factor_terms(sp.Add(*(x for x in sp.Add.make_args(sp.expand_mul(expr)) if x.free_symbols & variables)))


def module_normalize(original):
    """Gauge-fix a positive common amplitude only with an exact Euler certificate.

For a positive pair (a,b), if a*f_a+b*f_b has no input dependence,
the centred response is invariant under common scaling. Fix a=1 and
interpret the remaining b as b/a. This is not an arbitrary parameter fix.
Extra parameter inequalities are conservatively excluded from this rule.
"""
    row = copy.deepcopy(original); events = []
    while True:
        m = Module(row); variables = {m.symbols[v] for v in ('tau', 'c', 'h')}
        pairs = itertools.combinations([p for p in m.params if p['domain'] == 'POSITIVE'], 2)
        changed = False
        for left, right in pairs:
            a, b = m.symbols[left['name']], m.symbols[right['name']]
            # Cheap structural gate: do not ask a general simplifier to prove
            # identities for every nonlinear pair. Only a homogeneous linear
            # positive-rate sum is eligible for this exact gauge rule.
            homogeneous_sum = False
            for node in sp.preorder_traversal(m.expr):
                if not isinstance(node, sp.Add) or not {a,b} <= node.free_symbols: continue
                try:
                    poly = sp.Poly(node, a, b)
                    if all(sum(power) == 1 for power, _ in poly.terms()):
                        homogeneous_sum = True; break
                except sp.PolynomialError:
                    pass
            if not homogeneous_sum: continue
            # Homogeneous inequality residuals preserve their sign under a
            # positive common scaling, including interior-peak endpoint rules.
            safe_conditions = True
            for _, e, _, _ in m.conditions:
                if not e.free_symbols & {a,b}: continue
                try:
                    degrees = {sum(power) for power, _ in sp.Poly(e, a, b).terms()}
                    if len(degrees) != 1: safe_conditions = False; break
                except sp.PolynomialError:
                    safe_conditions = False; break
            if not safe_conditions: continue
            euler = sp.cancel(a * sp.diff(m.expr, a) + b * sp.diff(m.expr, b))
            if euler.free_symbols & variables:
                continue
            if sp.diff(m.expr, a) == 0 or sp.diff(m.expr, b) == 0:
                continue
            replacement = m.expr.subs(a, 1)
            p = row['proposal']; p['expression'] = str(replacement)
            p['parameters'] = [x for x in p['parameters'] if x['name'] != left['name']]
            for x in p['parameters']:
                if x['name'] == right['name']:
                    x['meaning'] = 'Identifiable ratio ' + right['name'] + '/' + left['name'] + '; ' + x['meaning']
            p['conditions'] = [dict(c, expression=str(e.subs(a, 1))) for c, e, _, _ in m.conditions
                               if e.subs(a, 1) != 1]
            events.append(dict(rule='EXACT_CENTRED_POSITIVE_SCALE_GAUGE', fixed=left['name'],
                               ratio=right['name'], euler_constant=str(euler),
                               original_expression=str(m.expr), new_expression=str(replacement),
                               numerical_guard_note='Same mathematical family on positive domains; finite solver boxes are guards, not a physical prior.'))
            changed = True; break
        if not changed: break
    if events:
        row['parent_module_id'] = original['module_id']
        row['module_id'] = original['module_id'] + '_G'
        row['exact_reparameterizations'] = events
        row['parent_checks_before_reparameterization'] = row.get('checks', {})
        m = Module(row)
        ref = {m.symbols['tau']:1, m.symbols['c']:sp.Rational(7,2), m.symbols['h']:1}
        row['checks'] = dict(exact_reparameterization_certificates=events,
            canonical_centered_expression=str(m.expr-m.expr.subs(ref)),
            structural_profile=generation.structural_profile(dict(row['proposal'],
                construction_family_id=row['proposal'].get('construction_family_id','EXACT_GAUGE'),
                mathematical_distinction=row['proposal'].get('mathematical_distinction','Same centred family, identifiable ratio')), {}))
    row['canonical_math_profile'] = math_profile(row)
    return row, events


def exact_linear_fusion(records):
    """Merge two unconstrained real linear coefficients with identical responses.

No fitted values enter the symbolic proof. Only exact column equality up to
an input-independent offset is used. Sharing unequal effects is NOT this rule.
"""
    if len(records) < 2: return None
    expression = 0; parameters = []; conditions = []; symbols = {}
    for index, row in enumerate(records):
        m = Module(row); mapping = {m.symbols[p['name']]: sp.Symbol('m%d_%s' % (index, p['name']),
                    **({'positive': True} if p['domain'] == 'POSITIVE' else {'real': True})) for p in m.params}
        expression += m.expr.xreplace(mapping)
        for p in m.params:
            name = str(mapping[m.symbols[p['name']]])
            parameters.append(dict(p, name=name)); symbols[name] = mapping[m.symbols[p['name']]]
        conditions += [dict(c, expression=str(e.xreplace(mapping))) for c, e, _, _ in m.conditions]
    variables = {sp.Symbol('tau', positive=True), sp.Symbol('c', nonnegative=True), sp.Symbol('h', positive=True)}
    proofs = []
    for left, right in itertools.combinations(list(parameters), 2):
        if left not in parameters or right not in parameters: continue
        if left['domain'] != 'REAL' or right['domain'] != 'REAL': continue
        a, b = symbols[left['name']], symbols[right['name']]
        # A constrained sum cannot be removed without transforming its contract.
        if any(a in construction.symbolic(c['expression'], {**{str(v): v for v in variables}, **symbols}).free_symbols or
               b in construction.symbolic(c['expression'], {**{str(v): v for v in variables}, **symbols}).free_symbols for c in conditions):
            continue
        if sp.diff(expression, a, 2) != 0 or sp.diff(expression, b, 2) != 0: continue
        residual = sp.diff(expression, a) - sp.diff(expression, b)
        if all(sp.simplify(sp.diff(residual, v)) == 0 for v in variables):
            expression = expression.subs(b, 0); parameters.remove(right)
            left['meaning'] = 'Identifiable sum of coefficients: ' + left['name'] + ' + ' + right['name']
            proofs.append(dict(rule='EXACT_IDENTICAL_REAL_LINEAR_COLUMN_MERGE', kept=left['name'],
                               removed=right['name'], derivative_difference=str(residual)))
    if not proofs: return None
    expression = variable_part(expression, variables)
    parents = sorted(r['module_id'] for r in records)
    mid = 'X_' + hashlib.sha256(json.dumps(parents).encode()).hexdigest()[:12]
    p = dict(proposal_id=mid, hypothesis_id='H4', expression=str(expression), parameters=parameters,
             conditions=conditions, family='Exact complete-equation identifiable reparameterization',
             construction_basis='EXACT_REPARAMETERIZATION', source_links=[],
             process_argument='No additional mechanism claim: remove exact centred response redundancy.',
             mathematical_distinction='Same identifiable family as parent complete equation.')
    row = dict(module_id=mid, proposal=p, assembly_scope='COMPLETE_EQUATION',
               parent_modules=parents, exact_reparameterizations=proofs)
    row['canonical_math_profile'] = math_profile(row)
    return row
