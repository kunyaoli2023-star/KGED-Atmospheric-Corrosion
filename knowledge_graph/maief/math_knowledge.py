"""Prebuilt mathematical knowledge graph and deterministic template compiler.

Scientific hypotheses are inputs, never conclusions certified by this graph.
"""
import ast
import copy
from pathlib import Path
from . import one_shot, operator_interface, operator_library
from .storage import read, fingerprint

ROOT=Path(__file__).resolve().parents[1]
VERSION='MATHEMATICAL_KNOWLEDGE_GRAPH_1'
FAMILIES={
 'exponential_power':'指数与幂律族', 'bell_decay':'峰形与衰减族',
 'rational':'有理函数与饱和族', 'logarithmic':'对数族',
 'special_saturation':'特殊函数与平滑饱和族', 'polynomial':'多项式对照族',
 'interaction':'多变量交互族', 'null':'零效应对照'}

def build_graph():
    registry=read(ROOT/'inputs/symbol_registry.json')
    protocol=read(ROOT/'inputs/module_protocol.json')
    hypotheses=read(ROOT/'inputs/hypotheses.json')['hypotheses']
    nodes={};edges=[]
    def node(id,kind,**fields):nodes[id]={'id':id,'kind':kind,**fields};return id
    def edge(a,r,b):edges.append({'source':a,'relation':r,'target':b})
    for name,spec in registry['raw_quantities'].items():node('raw:'+name,'raw_quantity',**spec)
    for name,spec in registry['variables'].items():
        assert spec['domain']==protocol['domains'][name] and spec['reference']==protocol['reference'][name]
        node('symbol:'+name,'coordinate',symbol=name,**spec)
        edge('raw:'+spec['raw'],'normalizes_to','symbol:'+name)
    for h in hypotheses:
        node('hypothesis:'+h['hypothesis_id'],'hypothesis',**h)
        for v in h['variables']:edge('hypothesis:'+h['hypothesis_id'],'uses_coordinate','symbol:'+v)
    for name,label in FAMILIES.items():node('family:'+name,'mathematical_family',label=label,
        status='Candidate structural family, not an established corrosion mechanism')
    for name,domain in [('POSITIVE','strictly greater than zero'),('REAL','finite real')]:
        node('parameter_type:'+name,'parameter_type',domain=domain,dimension='dimensionless',
             naming='p0,p1,... allocated per candidate',sharing='No fitted values shared between candidates')
    functions={
        'log':('strictly positive','real','log(x)'),
        'log1p':('greater than -1; nonnegative preferred for promotion','real','log(1+x)'),
        'exp':('real; overflow checked numerically','positive','exp(x)'),
        'sqrt':('nonnegative','nonnegative','sqrt(x)'),
        'tanh':('real','between -1 and 1','tanh(x)')}
    for name,(domain,out,definition) in functions.items():
        node('operator:'+name,'operator',input_domain=domain,output_domain=out,definition=definition)
    arithmetic={ast.Add:('add','real + real'),ast.Sub:('subtract','real - real'),
                ast.Mult:('multiply','real * real'),ast.Div:('divide','nonzero denominator; full domain and reference must be checked'),
                ast.Pow:('power','real exponent needs positive base; zero base with positive exponent allowed with boundary checks')}
    for name,domain in arithmetic.values():
        node('operator:'+name,'arithmetic_operator',input_domain=domain,
             output_domain='Depends on operands; type and numerical checks required')
    for name,spec in operator_library.CATALOG.items():
        node('operator:'+name,'protected_operator',**spec,epsilon=operator_library.EPSILON,
             policy='Explicit opt-in only; expansion, derivatives and numerical execution must agree')
    def template(hid,key,family,expression,meanings,trend='UNCONSTRAINED',boundary=''):
        tid='template:'+hid+':'+key
        params=[{'domain':'POSITIVE','meaning':m} for m in meanings]
        node(tid,'executable_template',hypothesis_id=hid,family_id='family:'+family,
             formula_py=expression,formula_kind='LOG_RESPONSE',parameters=params,
             expected_shape=trend,boundary_notes=boundary,
             parameter_bounds_policy='Positive domain only; numerical fit policy chooses transformed coordinates. Old dimensional bounds are not silently reused.',
             physical_status='Requires proposal-specific scientific justification',
             origin='Existing V6 construction or explicit mathematical family implementation; not a fitted winner')
        edge('hypothesis:'+hid,'allows_candidate',tid);edge('family:'+family,'has_template',tid)
        for name in sorted({n.id for n in ast.walk(ast.parse(expression,mode='eval')) if isinstance(n,ast.Name)}):
            if name in registry['variables']:edge(tid,'uses_coordinate','symbol:'+name)
            elif name in functions:edge(tid,'uses_operator','operator:'+name)
        for op in {type(n.op) for n in ast.walk(ast.parse(expression,mode='eval')) if isinstance(n,ast.BinOp)}:
            edge(tid,'uses_operator','operator:'+arithmetic[op][0])
        if params:edge(tid,'requires_parameter_type','parameter_type:POSITIVE')
    for hid,choices in one_shot.CATALOG.items():
        for key,(expr,meanings,trend) in choices.items():
            family=('null' if key=='null' else 'interaction' if hid=='H4' else
                {'activated':'exponential_power','linear':'polynomial','logarithmic':'logarithmic',
                 'saturation':'rational','competition':'bell_decay','transport_peak':'bell_decay','attenuation':'polynomial'}[key])
            template(hid,key,family,expr,meanings,trend)
    for hid,x in [('H1','tau'),('H2','c'),('H3','h')]:
        template(hid,'exponential_saturation','special_saturation',f'p0*(1-exp(-p1*{x}))',
                 ['response amplitude','positive saturation rate'],'INCREASING','Finite at zero; tends to p0 as the coordinate grows.')
        template(hid,'tanh_saturation','special_saturation',f'p0*tanh(p1*{x})',
                 ['response amplitude','positive saturation rate'],'INCREASING','Finite at zero; tends to p0.')
        template(hid,'power_response','exponential_power',f'p0*{x}**p1',
                 ['response amplitude','positive power exponent'],'INCREASING',
                 'Finite response at zero; derivative may be singular for exponent below one. No smoothness guarantee at zero.')
    for hid,x in [('H2','c'),('H3','h')]:
        template(hid,'gaussian_log_response','bell_decay',f'-({x}-p0)**2/p1**2',
                 ['positive peak coordinate','positive width'],'INTERIOR_MAXIMUM',
                 'Peak exists at p0 on the positive axis, not necessarily inside the measured interval.')
        template(hid,'bell_log1p','bell_decay',f'log1p(p0*{x}*exp(-p1*{x}))',
                 ['positive strength','positive decay rate'],'INTERIOR_MAXIMUM',
                 'Log-response closure with finite zero boundary; not an exact log of a zero-baseline rate.')
        template(hid,'power_bell_log1p','bell_decay',f'log1p(p0*{x}**p1*exp(-p2*{x}))',
                 ['positive strength','positive growth exponent','positive decay rate'],'INTERIOR_MAXIMUM',
                 'Peak coordinate p1/p2; may lie outside the measured interval; zero derivative may be singular.')
    for name in ['SUM_LOG_RESPONSES','DIFFERENCE_LOG_RESPONSES']:
        node('construction:'+name,'construction_rule',input='Two template IDs compatible with the target hypothesis',
             output='Log-response contribution; parameter names allocated independently',
             status='Implemented composition; each resulting candidate must pass domain and identifiability checks')
        for fid in FAMILIES:
            if fid!='null':edge('family:'+fid,'may_participate_in','construction:'+name)
    edges=sorted(edges,key=lambda e:(e['source'],e['relation'],e['target']))
    graph={'version':VERSION,'nodes':nodes,'edges':edges,'response_contract':protocol['response'],
        'validation_scope':'Finite probe checks and symbolic checks, not physical truth or complete identifiability proof',
        'protection_policy':'No silent clipping. Registered explicit protection is permitted. Older symbol-registry blanket epsilon prohibition is not imported into this graph.',
        'input_hashes':{n:fingerprint(read(ROOT/'inputs'/n)) for n in ['symbol_registry.json','module_protocol.json','hypotheses.json']},
        'extension_policy':'New templates require domain/parameter contracts, compilation and checks; graph revision is versioned. No automatic LLM repair.'}
    from .process_taxonomy import upgrade
    return upgrade(graph)

def compile_template(graph,template_id,scientific_argument):
    if not scientific_argument.strip():raise ValueError('SCIENTIFIC_ARGUMENT_REQUIRED')
    t=graph['nodes'][template_id]
    if t['kind']!='executable_template':raise ValueError('NOT_A_TEMPLATE')
    raw={'proposal_id':template_id,'hypothesis_id':t['hypothesis_id'],'family':t['family_id'],
         'formula_py':t['formula_py'],'formula_kind':t['formula_kind'],'parameters':copy.deepcopy(t['parameters']),
         'scientific_argument':scientific_argument,'shape_expectation':t['boundary_notes'] or t['expected_shape'],
         'expected_shape':t['expected_shape'],'source_links':[],
         'evidence_gap':'Graph entry is mathematical scaffolding; no literature verification implied.',
         'conditions':[],'construction_family_id':t['family_id'],'mathematical_distinction':template_id,
         'question_links':[{'hypothesis_id':t['hypothesis_id'],'relation':'RIVAL_CONTROL' if t['formula_py']=='0' else 'IMPLEMENTATION',
                            'observable_prediction':t['boundary_notes'] or t['expected_shape']}], 'contrast_links':[]}
    result=operator_interface.compile_operator(raw)
    result['interface_receipt']['math_knowledge_graph']={'sha256':graph['sha256'],'template_id':template_id}
    return result

def compile_combination(graph,left,right,operation,scientific_argument):
    if operation not in ('SUM_LOG_RESPONSES','DIFFERENCE_LOG_RESPONSES'):raise ValueError('UNKNOWN_CONSTRUCTION')
    a=graph['nodes'][left];b=graph['nodes'][right]
    if left==right or a['hypothesis_id']!=b['hypothesis_id']:raise ValueError('REQUIRE_DISTINCT_SAME_HYPOTHESIS_TEMPLATES')
    if a['kind']!='executable_template' or b['kind']!='executable_template':raise ValueError('NOT_A_TEMPLATE')
    offset=len(a['parameters'])
    class Rename(ast.NodeTransformer):
        def visit_Name(self,node):
            if node.id.startswith('p') and node.id[1:].isdigit():
                return ast.copy_location(ast.Name(id='p'+str(offset+int(node.id[1:])),ctx=ast.Load()),node)
            return node
    expression=ast.unparse(ast.fix_missing_locations(Rename().visit(ast.parse(b['formula_py'],mode='eval'))))
    work=copy.deepcopy(graph);id='template:composite:'+fingerprint([left,right,operation])[:16]
    work['nodes'][id]={**copy.deepcopy(a),'id':id,'formula_py':f"({a['formula_py']}){'+' if operation=='SUM_LOG_RESPONSES' else '-'}({expression})",
                       'parameters':a['parameters']+b['parameters'],'expected_shape':'UNCONSTRAINED',
                       'boundary_notes':'Composition requires its own shape and identifiability checks.'}
    result=compile_template(work,id,scientific_argument)
    result['interface_receipt']['math_knowledge_graph']['recipe']={'left':left,'right':right,'operation':operation}
    return result
