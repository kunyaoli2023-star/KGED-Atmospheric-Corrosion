"""Single-call family proposals and a typed, acyclic compositional grammar."""
import copy
from . import one_shot, operator_interface as compact
from .construction import obj, arr, STR, enum
from .storage import fingerprint

VERSION='TYPED_FAMILY_GRAPH_1'
OPS=['variable','parameter','one','add','multiply','divide','subtract','negative','log','log1p']
INSTRUCTIONS='''Propose multiple scientifically justified competing mathematical families
for the assigned hypothesis in ONE response. Aim to cover at least three distinct
non-null structures where justified; explicitly record coverage gaps otherwise.
Include a compositional challenger when scientifically justified, using construction_id
COMPOSE and an acyclic construction_nodes list. Existing family anchors can be selected
by their graph ID with construction_nodes=[]. Up to six proposals is a resource cap,
not a success quota. Do not pad with aliases, renamed coefficients or fixed exponent grids.
Each proposal explains differences in low-input behavior, saturation, tail, peak or
coupling through mathematical_distinction. New combinations are allowed, not arbitrary
formula strings. All outputs are dimensionless LOG_RESPONSE contributions.
Nodes are ordered: args reference only EARLIER node IDs. The LAST node is the output.
variable: args=[], variable=tau/c/h, parameter_meaning=''. parameter: args=[],
variable='', parameter_meaning=physical role (positive parameter allocated by compiler).
one: args=[], variable='', parameter_meaning=''. Other ops use variable='' and
parameter_meaning='', with args length 1 for negative/log/log1p, otherwise 2.
divide requires a strictly positive denominator; log requires strictly positive input;
log1p requires nonnegative input. add/multiply preserve positivity; subtract/negative
produce unconstrained real values. Do not hide invalid domains with guards or floors.
At most 18 nodes and four parameters per composition. Every node must reach the output.
H1 uses only tau; H2 only c; H3 only h; H4 may combine tau,c,h.
No mathematical repair loop. Unsupported proposals are reported, not rewritten.
Do not claim a novel scientific mechanism merely because a new combination is legal.
'''

def schema(hid):
    result=one_shot.schema(hid)
    item=result['properties']['proposals']['items']
    item['properties']['construction_id']['enum'].append('COMPOSE')
    node=obj({'id':STR,'op':enum(OPS),'args':arr(STR),
              'variable':enum(['','tau','c','h']),'parameter_meaning':STR})
    item['properties']['construction_nodes']={**arr(node),'maxItems':18}
    item['required'].append('construction_nodes')
    result['properties']['proposals']['maxItems']=6
    return result

def audit(records):
    """Summarize checked rows; structural diversity is not scientific novelty."""
    good=[r for r in records if r.get('math_pass') and r.get('canonical_key')]
    keys={r['canonical_key'] for r in good}
    anchors={r['formula'] for r in good if r['formula'] not in ('null','COMPOSE')}
    return {'generated':len(records),'math_pass':len(good),'unique_centered_formulas':len(keys),
        'duplicate_valid_formulas':len(good)-len(keys),'anchor_families':sorted(anchors),
        'valid_compositions':sum(r['formula']=='COMPOSE' for r in good),
        'null_controls':sum(r['formula']=='null' for r in good),
        'scope':'Compiler/checker coverage only. Family names are not proof of novelty; compositions may equal known anchors.'}

def prepare_payload(payload):
    result=one_shot.prepare_payload(payload)
    result['shared_symbol_knowledge']['reusable_constructions']=copy.deepcopy(
        payload.get('shared_symbol_knowledge',{}).get('reusable_constructions',[]))
    result['instructions']=INSTRUCTIONS
    result['family_construction_contract']={'version':VERSION,'operators':OPS,
        'sign_types':['POSITIVE','NONNEGATIVE','REAL'],'parameter_limit':4,
        'policy':'Finite grammar with extensible compositions; no post-generation LLM repair.',
        'novelty':'Compared by centered canonical formula, not by model-assigned family name.'}
    return result

def compile_operator(raw):
    if raw['construction_id']!='COMPOSE':
        if raw.get('construction_nodes'):raise ValueError('ANCHOR_MUST_NOT_HAVE_NODES')
        return one_shot.compile_operator(raw)
    nodes=raw['construction_nodes']
    if not 1<=len(nodes)<=18:raise ValueError('NODE_BUDGET')
    allowed={'H1':{'tau'},'H2':{'c'},'H3':{'h'},'H4':{'tau','c','h'}}[raw['hypothesis_id']]
    expressions={};signs={};ancestors={};parameters=[]
    for node in nodes:
        key=node['id'];op=node['op'];args=node['args']
        if not key or key in expressions or op not in OPS:raise ValueError('INVALID_NODE')
        arity=0 if op in ['variable','parameter','one'] else 1 if op in ['negative','log','log1p'] else 2
        if len(args)!=arity or any(a not in expressions for a in args):raise ValueError('FORWARD_REFERENCE_OR_ARITY')
        # Fixed operation semantics determine which fields are executable.
        # Descriptive metadata on other fields is retained in the receipt, not code.
        args_expr=[expressions[a] for a in args];types=[signs[a] for a in args]
        if op=='variable':
            if node['variable'] not in allowed:raise ValueError('CROSS_HYPOTHESIS_VARIABLE')
            expression=node['variable'];sign='NONNEGATIVE' if expression=='c' else 'POSITIVE'
        elif op=='parameter':
            if not node['parameter_meaning'].strip() or len(parameters)>=4:raise ValueError('PARAMETER_CONTRACT')
            expression='p'+str(len(parameters));sign='POSITIVE'
            parameters.append({'domain':'POSITIVE','meaning':node['parameter_meaning']})
        elif op=='one':expression='1';sign='POSITIVE'
        elif op in ['log','log1p']:
            if types[0] not in (['POSITIVE'] if op=='log' else ['POSITIVE','NONNEGATIVE']):raise ValueError('DOMAIN_TYPE: '+op)
            expression=f'{op}({args_expr[0]})';sign='REAL' if op=='log' else types[0]
        elif op=='negative':expression=f'-({args_expr[0]})';sign='REAL'
        else:
            if op=='divide' and types[1]!='POSITIVE':raise ValueError('DOMAIN_TYPE: denominator')
            token={'add':'+','subtract':'-','multiply':'*','divide':'/'}[op]
            expression=f'({args_expr[0]}){token}({args_expr[1]})'
            sign='REAL'
            if op!='subtract' and 'REAL' not in types:
                sign='POSITIVE' if (op=='add' and 'POSITIVE' in types) or all(t=='POSITIVE' for t in types) else 'NONNEGATIVE'
        expressions[key]=expression;signs[key]=sign
        ancestors[key]={key}.union(*(ancestors[a] for a in args))
    if ancestors[nodes[-1]['id']]!=set(expressions):raise ValueError('UNUSED_NODES')
    candidate=copy.deepcopy(raw)
    candidate.update(formula_py=expressions[nodes[-1]['id']],parameters=parameters,
        formula_kind='LOG_RESPONSE',conditions=[],expected_shape='UNCONSTRAINED',
        shape_expectation='Composite shape must be evaluated; no automatic monotonicity or peak claim.')
    result=compact.compile_operator(candidate)
    result['interface_receipt']['construction_graph']={'version':VERSION,'id':'COMPOSE',
        'definition_sha256':fingerprint(nodes),'nodes':nodes,'inferred_signs':signs,
        'formula_generated_by':'typed acyclic graph compiler'}
    return result
