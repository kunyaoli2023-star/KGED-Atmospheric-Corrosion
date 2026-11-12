"""Construction schema, scientific instructions, independent checks."""
import ast, copy, json, re
import sympy as sp
from . import construction as base
HIDS=["H1","H2","H3","H4"]


def namespace_proposals(proposals, batch, repair, hid):
    """Local transport labels are not scientific identifiers or evidence IDs.

    The original reply stays archived verbatim. Every list occurrence receives
    the same deterministic batch/slot/index prefix used for unique replies.
    No expression, conditions, source links, or scientific decisions change.
    Scientific duplicate handling happens only after mathematical checking.
    """
    if any(p['hypothesis_id'] != hid for p in proposals):
        raise RuntimeError('PROPOSAL_CROSSED_DECLARED_QUESTION_SCOPE')
    counts = {}
    result, records = [], []
    for index, original in enumerate(proposals):
        raw_id = original['proposal_id']
        counts[raw_id] = counts.get(raw_id, 0) + 1
        proposal = copy.deepcopy(original)
        proposal['proposal_id'] = f'{batch}_{repair}_{hid}_{index}::' + raw_id
        result.append(proposal)
        records.append(dict(reply_index=index, model_proposal_id=raw_id,
                            internal_proposal_id=proposal['proposal_id']))
    return result, dict(policy='DETERMINISTIC_BATCH_SLOT_REPLY_INDEX_NAMESPACE',
        original_reply_unchanged=True, mathematical_content_changed=False,
        scientific_gates_waived=False,
        duplicate_model_proposal_ids=sorted(name for name, count in counts.items() if count > 1),
        records=records)


def construction_focus(hid, cycle_index, admitted_rows, plan):
    """Observable, repeatable construction challenge; never a form quota.

    Family declarations are hypotheses about useful routes, not evidence that
    the resulting equations are mathematically different or physically true.
    The independent structural fingerprint and admitted mathematical key are
    therefore reported separately from the prose family labels.
    """
    families = sorted((item for item in plan.get('construction_families', [])
                       if item.get('hypothesis_id') == hid),
                      key=lambda item: item['construction_family_id'])
    rows = [row for row in admitted_rows if row.get('proposal', {}).get('hypothesis_id') == hid]
    admitted_families = sorted({row['proposal'].get('construction_family_id', '') for row in rows})
    distinct_keys = sorted({row.get('original_checks', row.get('checks', {})).get('dedup_key', '')
                            for row in rows} - {''})
    structural = sorted({row.get('original_checks', row.get('checks', {}))
                         .get('structural_profile', {}).get('structural_fingerprint', '')
                         for row in rows} - {''})
    missing = [item['construction_family_id'] for item in families
               if item['construction_family_id'] not in admitted_families]
    target = (missing[cycle_index % len(missing)] if missing else
              families[cycle_index % len(families)]['construction_family_id'] if families else None)
    return {'policy':'EVIDENCE_CONDITIONED_CONSTRUCTION_FOCUS_V12',
            'focus_id':base.sha(json.dumps([hid, cycle_index, target, distinct_keys, structural],
                ensure_ascii=False, sort_keys=True).encode())[:16],
            'hypothesis_id':hid, 'cycle_index':cycle_index,
            'priority_family_id':target,
            'unrepresented_declared_family_ids':missing,
            'admitted_module_count':len(rows),
            'distinct_checked_mathematical_keys':len(distinct_keys),
            'distinct_syntactic_profiles':len(structural),
            'interpretation':'Priority is a challenge, not an allowed-form list, a quota, a mechanism proof or a reason to stop. Propose other justified alternatives and null rivals when warranted.',
            'no_target_module_or_coefficient_count':True}

MINIMAL_CONSTRUCTION = """
For every justified process topology, consider its simplest interpretable
member before adding freely varying orders, background terms or interactions.
Retain simple members alongside meaningful extensions when both are justified;
there is no module-count or coefficient-count quota. Distinguish positive
capacities from resistances: parallel rates add, serial resistances add, and
a product closure needs its own assumption. Do not label these as equivalent.
A generalized transport or establishment order should have a first-order rival
when interpretable under stated geometry, steady-state and property assumptions.
This is a proposed closure, not a requirement to reproduce a published equation.
Conditions apply throughout the predictor domain and at the reference point:
do not encode one side of an interior peak as a global fitting condition.
Do not decorate simple laws with arbitrary powers to inflate candidate counts.
"""
COMMON = "Construct scientifically motivated mathematical MODULES, not a final equation.\nNo target total, per-hypothesis quota or fixed coefficient count. A process is NOT a unique curve:\nexplore inequivalent response laws when curvature, low-variable order, attenuation tail, peak\nsymmetry or coupling makes a testable difference. Do not stop merely because each trend has an example.\nDo not enumerate arbitrary operator combinations, tune fixed numerical powers just to inflate a list,\nor repeat a family by parameter renaming/centering constants. General shape parameters are allowed\nwhen interpretable and constrained; do not pretend several fixed powers are independent mechanisms.\nUse tau,c,h and declared parameters only; arithmetic, **, log,log1p,exp,sqrt,tanh. All coordinates\nand parameters are dimensionless. H1 tau-only, H2 c-only, H3 h-only, H4 h plus environment or an\nexplicit invariant rival. h is LIQUID electrolyte thickness, never rust/oxide/metal loss.\nOutput is a component of log current, centred later; a raw current/flux relation is not already\na log-current term. A chosen log-response closure is permitted as PHENOMENOLOGICAL_CLOSURE;\ndeclare its marginal-response/competition assumption. It NEED NOT be a published exact formula.\nFor SOURCE_LOG_TRANSFORM/REDUCED_PROCESS_BALANCE first state an independently motivated positive\nrate expression, then derive its logarithm. rate=exp(already chosen f) is not independent derivation.\nrate_expression_before_log accepts executable RHS ONLY, never R(c)=... or prose.\nOmit a positive constant rate baseline that cancels from log-response centering, and do not\ndeclare it as a learned parameter. Leave this field empty for a chosen phenomenological log\nclosure or a rival that does not claim an independent rate-to-log derivation.\nAnalogy from a different material/response is explicitly qualified, not called a direct source law.\nFor LOCAL_APPROXIMATION identify the parent relation, expansion point and small-perturbation scope.\nUse scientific_role MECHANISM_EXTENSION for a new analogy/closure, RIVAL_CONTROL for a rival;\nTHEORY_REEXPRESSION requires genuine source-dependent algebra, not just a related paper.\nSource IDs must be exact supplied short IDs. Sources support only the physical statement actually\nin their extraction. Empty source_links is allowed only with an honest evidence_gap and a coherent\nproposed modelling assumption. No fake pages, no literature-proven peaks inferred from 'chloride'.\nEvery proposal includes an ordered construction, final executable expression, parameter meanings,\nsufficient sign/domain conditions, correct derivative, limits with response scale, discriminating\nprediction and falsification. Claimed derivative is executable RHS ONLY. Last construction step\nequals expression EXACTLY. Conditions are arithmetic residuals against zero using GT_ZERO/GE_ZERO,\nnot comparisons in expression strings. State conditions for peaks, not that they must lie in the\nmeasured interval. Distinguish a finite plateau from a decreasing slope of an unbounded log law.\nSimple uncoupled/null rivals are admissible; no free parameter removed by reference centering.\nFor H4 nonzero mixed h-environment derivatives establish coupling, not proof of a physical mechanism.\nMatch construction_family_id to a supplied plan or propose a justified new one. mathematical_distinction\nmust identify the law/topology/tail/coupling distinction, not merely give a different family name.\nChinese explanations, executable ASCII expressions. Return every genuinely justified distinct\nconstruction you can defend in this call; do not deliberately provide a minimal representative list.\n"

COMMON += """
SYMBOL COMPILER CONTRACT (supersedes the older all-fields-final-symbols rule):
Consult shared_symbol_knowledge before proposing symbols. For the same recorded definition,
mathematical role, meaning and conditions, reuse canonical_symbol and preserve its meaning/domain.
Aliases in prior responses are historical names, not new independent mechanisms. A new meaning
requires a separate declaration. The graph's parameter concepts never share fitted values across
candidates. Explicitly redeclare any reused intermediate alias in symbol_definitions so its mapping
is checked again; do not treat graph membership as permission to bypass mathematical review.
source_symbols records original notation, meaning, unit and mapping_note; it never declares fit parameters.
symbol_definitions records explicit acyclic intermediate aliases with name, expression and meaning.
Every executable alias must expand to tau,c,h, declared fitted parameters and numeric constants.
Final expression uses only tau,c,h and actual fitted parameters. Rate, derivative, construction steps
and conditions may use declared aliases, which are expanded and rechecked before fitting.
For example with final parameter a: define Ea_bar=a and inv_tau=1/tau, then the final expression
is a*(1-1/tau). Explain Ea/(R*T_ref)=a in source_symbols.mapping_note; do not leave Ea,R,T_ref
unmapped in executable fields. A source relation without an executable mapping belongs in statement
or governing_relation prose. A k0 prefactor may be normalized to 1 only with an explicit explanation
that it is positive, coordinate-independent and removed by log centering. Never fit it redundantly.
Do not use arbitrary undefined f(tau). Write its concrete approximation or keep the abstract function
in prose. Empty source_symbols and symbol_definitions arrays are valid for direct closures.
conditions are GLOBAL feasibility requirements enforced on data and reference points. Local increasing
branches, peak equations and interval-dependent derivative signs belong in trend_scope, not conditions.
question_links must explicitly link each module to its supplied scientific question and observable
prediction or rival. A coupled H4 module may also address H3; do not confuse slot assignment with
physical behavior. contrast_links may use ONLY canonical registered Q_ IDs supplied in
new_subhypotheses. Empty contrast_links is valid when no registered contrast is being tested.
For every link distinguish IMPLEMENTATION from RIVAL_CONTROL and state the observable contrast.
Executable mathematical actions in feedback are tests, not instructions to reproduce a winner.
future_experiments are not available observations and cannot be claimed as completed tests.
"""


def schemas():
    bridge, proposals, review = base.schemas()
    family = base.obj({'construction_family_id': base.STR, 'hypothesis_id': base.enum(HIDS),
        'process_or_closure_assumption': base.STR, 'scientific_basis_and_limits': base.STR,
        'mathematical_distinction_to_test': base.STR, 'evidence_ids': base.arr(base.STR)})
    planning = base.obj({'bridges': bridge['properties']['bridges'],
        'overall_evidence_limits': base.STR, 'construction_families': base.arr(family),
        'unjustified_routes': base.arr(base.STR)})
    item = proposals['properties']['proposals']['items']
    item['properties'].update({'construction_family_id': base.STR,
                               'mathematical_distinction': base.STR,
        'question_links': base.arr(base.obj({'hypothesis_id':base.enum(HIDS),
            'relation':base.enum(['IMPLEMENTATION','RIVAL_CONTROL']), 'observable_prediction':base.STR})),
        'contrast_links': base.arr(base.obj({'contrast_id':base.STR,
            'relation':base.enum(['IMPLEMENTATION','RIVAL_CONTROL']), 'observable_prediction':base.STR}))})
    item['required'] += ['construction_family_id', 'mathematical_distinction','question_links','contrast_links']
    return planning, proposals, review, family


PROPOSAL_SLOT_CONTRACT = 'HYPOTHESIS_SCOPED_PROPOSAL_V1'


def proposal_schema_for(hid):
    """Bind module ownership to this call, without restricting cross-question links.

    A module belongs to one H1-H4 slot. Its question_links may still describe
    legitimate relationships to other scientific questions, especially H4/H3.
    No expression, parameter-count or proposal-count quota is imposed here.
    """
    if hid not in HIDS:
        raise ValueError('UNKNOWN_PROPOSAL_HYPOTHESIS')
    _, schema, _, _ = schemas()
    schema['properties']['proposals']['items']['properties']['hypothesis_id'] = base.enum([hid])
    return schema


def proposal_scope_violations(proposals, hid):
    """Return only transport ownership errors; scientific merit is checked later."""
    if hid not in HIDS:
        raise ValueError('UNKNOWN_PROPOSAL_HYPOTHESIS')
    return [{'index':index, 'declared_hypothesis_id':proposal.get('hypothesis_id')}
            for index,proposal in enumerate(proposals)
            if proposal.get('hypothesis_id') != hid]


def is_scope_schema_error(error):
    """Only a failed per-proposal ownership enum may be deferred as scope."""
    return bool(re.fullmatch(r'MODEL_SCHEMA_ERROR_AT_proposals\.\d+\.hypothesis_id', str(error)))


def check_links(proposal, contrasts=()):
    """References/scopes are hard checked; scientific adequacy still needs review."""
    questions={q['hypothesis_id']:q for q in proposal.get('question_links', [])}
    errors=[]
    if proposal['hypothesis_id'] not in questions:
        errors.append('Explicit question link for the assigned module slot is required.')
    if len(questions)!=len(proposal.get('question_links', [])):
        errors.append('Duplicate question links are not allowed.')
    names={n.id for n in ast.walk(ast.parse(proposal['expression'],mode='eval')) if isinstance(n,ast.Name)}
    for hid,link in questions.items():
        if not link['observable_prediction'].strip(): errors.append('Question link needs an observable prediction.')
        axis={'H1':'tau','H2':'c','H3':'h','H4':'h'}[hid]
        if link['relation']=='IMPLEMENTATION' and axis not in names:
            errors.append('Question implementation lacks its observable coordinate: '+hid)
    known={h['new_hypothesis_id']:h for h in contrasts}
    seen=set()
    for link in proposal.get('contrast_links', []):
        cid=link['contrast_id']
        if cid in seen: errors.append('Duplicate contrast link: '+cid)
        seen.add(cid)
        if cid not in known: errors.append('Unregistered contrast ID: '+cid)
        elif known[cid]['parent_hypothesis_id'] not in questions:
            errors.append('Contrast link requires an explicit parent question correspondence: '+cid)
        if not link['observable_prediction'].strip(): errors.append('Contrast link needs an observable prediction.')
    return errors


def structural_profile(p, check):
    """A structural descriptor, not an inequivalent-family proof or physics score."""
    tree=ast.parse(p['expression'],mode='eval')
    names=[n.id for n in ast.walk(tree) if isinstance(n,ast.Name)]
    params={x['name']:x for x in p['parameters']}
    order=list(dict.fromkeys(n for n in names if n in params))
    rename={n:'p'+str(i) for i,n in enumerate(order)}
    def encode(n):
        if isinstance(n,ast.Name): return rename.get(n.id,n.id)
        if isinstance(n,ast.Constant): return ('constant',n.value)
        if isinstance(n,ast.Call): return (n.func.id,encode(n.args[0]))
        if isinstance(n,ast.BinOp):return (type(n.op).__name__,encode(n.left),encode(n.right))
        if isinstance(n,ast.UnaryOp):return (type(n.op).__name__,encode(n.operand))
        if isinstance(n,ast.Expression):return encode(n.body)
        raise ValueError('Unexpected safe-tree node')
    fingerprint=base.sha(json.dumps([p['hypothesis_id'],encode(tree),
                         [params[n]['domain'] for n in order]]).encode())
    functions=sorted({n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call)})
    return {'structural_fingerprint':fingerprint, 'functions':functions,
        'operators':check.get('expression_operators'), 'parameters':len(params),
        'declared_construction_family':p['construction_family_id'],
        'mathematical_distinction':p['mathematical_distinction'],
        'scope':'Syntactic descriptor only; family counts and unequal fingerprints are not proofs of scientific independence.'}


def verify_extended(p,protocol,known):
    check=base.verify_one(p,protocol,set(known))
    if check['errors']: return check
    domains={x['name']: x['domain'] for x in p['parameters']}
    symbols={v:sp.Symbol(v,positive=True) for v in ['tau','h']}
    symbols['c']=sp.Symbol('c',nonnegative=True)
    symbols.update({n:sp.Symbol(n,**({'positive':True} if d=='POSITIVE' else {'real':True}))
                    for n,d in domains.items()})
    expr=base.symbolic(p['expression'],symbols)
    centred=expr-expr.subs({symbols[v]:x for v,x in protocol['reference'].items()},simultaneous=True)
    lost=[n for n in domains if sp.simplify(sp.diff(centred,symbols[n]))==0]
    check['parameters_removed_by_centering']=lost
    if lost:check['errors'].append('Parameters disappear after centering: '+','.join(lost))
    check['structural_profile']=structural_profile(p,check)
    check['status']='QUARANTINE' if check['errors'] else 'EXECUTABLE_WITH_CHECKED_CONSTRUCTION'
    return check
