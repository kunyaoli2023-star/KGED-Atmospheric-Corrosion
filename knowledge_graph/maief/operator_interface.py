"""V6 compact operator interface: science in prose, computation in a small DSL."""
import ast
import copy
import re
import sympy as sp
from . import construction as b
from .symbols import variable_symbols
from . import operator_library as library

INSTRUCTIONS = '''Generate mathematical operators for the assigned H1-H4 question.
Use formula_py with x for H1/H2/H3; the assigned tau/c/h is also accepted as an exact
alias. H4 uses tau,c,h explicitly and never an ambiguous x. All coordinates are
dimensionless with the supplied protocol domains. Parameter list order defines p0,p1,...;
An optional parameter symbol explicitly binds a local alias to its list index; no
symbol-to-parameter mapping is guessed. Never put undeclared Ea,R,k0 or functions in formula_py.
Allowed operations: +,-,*,/,**,log,log1p,exp,sqrt,tanh. Matching np.function spellings are
normalized by the compiler; arbitrary attribute access and undeclared functions are forbidden.
Explicit domain operators are safe_log_pos(z), safe_div_pos(a,b), positive_offset(z).
They expand respectively to log(z+epsilon), a/(b+epsilon), z+epsilon, with fixed
dimensionless epsilon=1e-8 and enforced nonnegative z/b. Explain their use in the
scientific argument; these are regularized models, not exact identities. Never use
them to hide negative rates. For POSITIVE_RATE at c=0, positive_offset(rate) is an
explicit option when a numerical rate floor is justified; otherwise propose a
scientifically justified positive background. Raw division/log remain unprotected.
Match parameter signs to shape claims: a REAL coefficient cannot guarantee increasing
p0*log1p(c). Do not impose local peak/branch conditions as global feasibility conditions.
formula_kind LOG_RESPONSE means an explicitly proposed contribution to log current;
POSITIVE_RATE means an independently motivated positive rate factor, whose logarithm
the program computes. These are scientifically different claims: explain the choice.
Describe sources, derivation, approximations, dimensions and physical parameter meaning
in scientific_argument and source_links. Do not generate executable derivation steps or
derivatives: the compiler generates these. Use only actual fitted parameters, never
constant prefactors that disappear under centering. New mathematical forms are allowed;
there is no fixed library, winner, formula family quota, or prescribed coefficient count.
conditions contain GLOBAL feasibility constraints, not a local increasing branch;
describe peak/branch conditions in shape_expectation. c includes zero; a rate used inside
log must remain positive. Query shared_symbol_knowledge for previously checked concepts,
but keep executable local names p0,p1,...; the graph stores physical names and aliases.
Repair only supplied failures on repair passes, preserving explicit scientific justification.
An empty proposal list must explain why. Preserve question/contrast links and evidence gaps.
'''


def schema(hid):
    from .generation import schemas
    _, full, _, _ = schemas()
    props = full['properties']['proposals']['items']['properties']
    parameter=b.obj({'domain':b.enum(['POSITIVE','REAL']), 'meaning':b.STR})
    parameter['properties']['symbol']=b.STR
    item=b.obj({'proposal_id':b.STR,'hypothesis_id':b.enum([hid]),'family':b.STR,
                'formula_py':b.STR,'formula_kind':b.enum(['LOG_RESPONSE','POSITIVE_RATE']),
                'parameters':b.arr(parameter),
                'scientific_argument':b.STR,'shape_expectation':b.STR,
                'expected_shape':b.enum(['INCREASING','DECREASING','INTERIOR_MAXIMUM','INTERIOR_MINIMUM','CONSTANT','UNCONSTRAINED']),
                'source_links':props['source_links'],'evidence_gap':b.STR,
                'conditions':props['conditions'],'construction_family_id':b.STR,
                'mathematical_distinction':b.STR,'question_links':props['question_links'],
                'contrast_links':props['contrast_links']})
    return b.obj({'proposals':b.arr(item),'coverage_not_claimed':b.arr(b.STR),'why_stop_adding_forms':b.STR})


def prepare_payload(payload):
    """Remove obsolete executable-symbol instructions from the model-facing context."""
    result=copy.deepcopy(payload)
    result['operator_interface']='V6_CANONICAL_OPERATOR_3'
    result['operator_library']={'version':library.VERSION,'epsilon':library.EPSILON,
                                'operators':copy.deepcopy(library.CATALOG)}
    protocol=result['protocol']
    protocol.pop('symbol_registry',None)
    result['executable_contract']={
        'variables':{'H1':['x','tau'],'H2':['x','c'],'H3':['x','h'],'H4':['tau','c','h']},
        'parameters':'List order defines p0,p1,...; optional symbol binds an explicit alias to that index.',
        'functions':list(b.FUNCTIONS)+list(library.CATALOG),
        'automatic':['coordinate alias mapping','declared parameter alias mapping','whitelisted np.function spelling','derivative calculation'],
        'never_automatic':['inventing a symbol meaning','changing a scientific law','clipping inputs','weakening constraints']}
    graph=result.get('shared_symbol_knowledge',{})
    graph['rules']=['Graph canonical symbols identify concepts; executable parameters use local p0,p1,... or explicitly declared symbol aliases.',
                    'Copy matching meaning/domain, but never share fitted values across candidates.',
                    'Redeclare all used parameters. Do not output symbol_definitions; this interface has no executable source derivation fields.']
    # Rich internal check records are evidence, not the schema to imitate on repair.
    repairs=[]
    for record in result.get('repair_records',[]):
        p=record['proposal']
        repairs.append({'proposal_id':p['proposal_id'],
                        'formula_py':p.get('formula_py',p.get('expression')),
                        'parameters':p.get('parameters',[]),
                        'errors':record.get('checks',{}).get('errors',[]),
                        'scientific_issues':record.get('scientific_review',{}).get('critical_issues',[]),
                        'instruction':'Return the compact operator schema. Correct mathematical issues; known coordinate aliases are already handled by the compiler.'})
    result['repair_records']=repairs
    old=result.pop('mathematical_repair_contract',{})
    result['prior_failure_evidence']=[{k:r.get(k) for k in ('hypothesis_id','expression','errors')}
                                      for r in old.get('previous_failures',[])]
    return result


def compile_operator(raw, *, epsilon=library.EPSILON):
    hid=raw['hypothesis_id'];axis={'H1':'tau','H2':'c','H3':'h','H4':'h'}[hid]
    parameters=[{'name':f'p{i}','domain':p['domain'],'meaning':p['meaning'],'dimension':'dimensionless'}
                for i,p in enumerate(raw['parameters'])]
    names={p['name'] for p in parameters}
    aliases = {}
    for index, parameter in enumerate(raw['parameters']):
        alias = parameter.get('symbol', f'p{index}')
        if (not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', alias) or alias in
                {'x','tau','c','h','np','T_C','C_wt','h_um'} | set(b.FUNCTIONS) | set(library.CATALOG)
                or alias in aliases or (re.fullmatch(r'p\d+',alias) and alias != f'p{index}')):
            raise ValueError('AMBIGUOUS_PARAMETER_ALIAS: '+str(alias))
        aliases[alias]=f'p{index}'
    changes=[]
    obligations=[]
    protections=[]
    def normalize(text):
        tree=ast.parse(text,mode='eval')
        class Functions(ast.NodeTransformer):
            def visit_Call(self,node):
                if (isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name)
                        and node.func.value.id=='np' and node.func.attr in b.FUNCTIONS):
                    changes.append({'from':'np.'+node.func.attr,'to':node.func.attr})
                    node.func=ast.copy_location(ast.Name(id=node.func.attr,ctx=ast.Load()),node.func)
                return self.generic_visit(node)
        tree=Functions().visit(tree)
        tree, required, expanded=library.lower(tree,epsilon)
        obligations.extend(required)
        protections.extend(expanded)
        tree=b.safe_tree(ast.unparse(tree),names|set(aliases)|({'x'} if hid!='H4' else set()))
        used={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}
        if hid!='H4' and used & ({'tau','c','h'}-{axis}):
            raise ValueError('CROSS_HYPOTHESIS_VARIABLE: '+','.join(sorted(used & ({'tau','c','h'}-{axis}))))
        class Replace(ast.NodeTransformer):
            def visit_Name(self,node):
                target=axis if node.id=='x' else aliases.get(node.id,node.id)
                if target!=node.id:changes.append({'from':node.id,'to':target})
                return ast.copy_location(ast.Name(id=target,ctx=ast.Load()),node)
        return ast.unparse(ast.fix_missing_locations(Replace().visit(tree)))
    formula=normalize(raw['formula_py'])
    conditions=[{**c,'expression':normalize(c['expression'])} for c in raw['conditions']]
    conditions += [{**c,'expression':normalize(c['expression'])} for c in list(obligations)]
    # Drop only syntactically unused declarations; no fitted or centered parameter
    # is silently removed. Keep original numbering to preserve explicit bindings.
    used={n.id for text in [formula]+[c['expression'] for c in conditions]
          for n in ast.walk(ast.parse(text,mode='eval')) if isinstance(n,ast.Name)}
    removed=[p['name'] for p in parameters if p['name'] not in used]
    parameters=[p for p in parameters if p['name'] in used]
    names={p['name'] for p in parameters}
    expression=f'log({formula})' if raw['formula_kind']=='POSITIVE_RATE' else formula
    symbols=variable_symbols()
    symbols.update({p['name']:sp.Symbol(p['name'],**({'positive':True} if p['domain']=='POSITIVE' else {'real':True})) for p in parameters})
    derivative=str(sp.diff(b.symbolic(expression,symbols),symbols[axis]))
    # Sympy may emit functions outside the declared grammar; never execute them unchecked.
    b.safe_tree(derivative,names)
    source_rate=raw['formula_kind']=='POSITIVE_RATE'
    rival=all(link['relation']=='RIVAL_CONTROL' for link in raw['question_links'])
    return {'proposal_id':raw['proposal_id'],'hypothesis_id':hid,
            'scientific_role':'RIVAL_CONTROL' if rival else 'MECHANISM_EXTENSION',
            'family':raw['family'],'expression':expression,
            'construction_basis':'REDUCED_PROCESS_BALANCE' if source_rate else 'PHENOMENOLOGICAL_CLOSURE',
            'governing_relation':raw['scientific_argument'],
            'rate_expression_before_log':formula if source_rate else '',
            'parameters':parameters,'source_symbols':[],'symbol_definitions':[],
            'interface_receipt':{'version':'V6_CANONICAL_OPERATOR_3','assigned_axis':axis,
                                 'parameter_aliases':aliases,'normalizations':changes,
                                 'formula_before':raw['formula_py'],'formula_after':formula,
                                 'scientific_form_changed':False,
                                 'removed_unused_declarations':removed,
                                 'operator_library':library.VERSION,'regularization':protections,
                                 'regularized_model':bool(protections)},
            'source_links':copy.deepcopy(raw['source_links']),'evidence_gap':raw['evidence_gap'],
            'process_argument':raw['scientific_argument'],
            'construction_steps':[{'operation':'EXACT_TRANSFORM' if source_rate else 'PROPOSED_CLOSURE',
                                   'statement':'Compiler-generated log transform of the supplied rate.' if source_rate else 'Compiler transcription of the explicitly proposed log-response closure.',
                                   'expression_after_step':expression}],
            'added_assumptions':[raw['scientific_argument']],
            'dimension_argument':'Coordinates and coefficients use the fixed dimensionless interface; scientific source mapping is reviewed separately.',
            'conditions':conditions,'derivative_axis':axis,'claimed_derivative':derivative,
            'trend':raw['expected_shape'],'trend_scope':raw['shape_expectation'],
            'limit_argument':'Boundary behavior checked numerically on the declared domain; no global asymptotic proof claimed.',
            'discriminating_prediction':raw['shape_expectation'],
            'falsification':'Compare the proposed shape and rival predictions against numerical diagnostics.',
            'novelty_relative_to_alternatives':raw['mathematical_distinction'],
            **{k:copy.deepcopy(raw[k]) for k in ('construction_family_id','mathematical_distinction','question_links','contrast_links')}}
