"""One scientific selection, deterministic construction, no mathematical repair.

This is a deliberately bounded initial construction graph, not free-form discovery.
All expressions are contributions to log current, not unqualified rate laws.
"""
import copy
from . import operator_interface as compact
from .storage import fingerprint

VERSION='PREDEFINED_CONSTRUCTION_GRAPH_1'
# Formula, positive parameter meanings, checked trend. h and tau are strictly positive.
CATALOG={
 'H1':{
  'activated':('p0*(1-1/tau)',['dimensionless activation strength'],'INCREASING'),
  'linear':('p0*(tau-1)',['positive temperature slope'],'INCREASING'),
  'logarithmic':('p0*log(tau)',['positive temperature elasticity'],'INCREASING')},
 'H2':{
  'saturation':('p0*c/(p1+c)',['promotion strength','positive saturation scale'],'INCREASING'),
  'logarithmic':('p0*log1p(c)',['positive chloride response strength'],'INCREASING'),
  'competition':('p0*c/(p1+c)-p2*c',['promotion strength','positive saturation scale','attenuation strength'],'UNCONSTRAINED')},
 'H3':{
  'transport_peak':('log(h)-log(p0+h**2)',['positive squared film scale'],'INTERIOR_MAXIMUM'),
  'attenuation':('-p0*h',['positive attenuation strength'],'DECREASING'),
  'saturation':('p0*h/(p1+h)',['film response amplitude','positive film scale'],'INCREASING')},
 'H4':{
  'salt_film':('p0*c*h/(1+p1*h)',['salt-film coupling strength','positive film attenuation'],'INCREASING'),
  'salt_film_log':('log1p(p0*c*h)',['positive salt-film coupling'],'INCREASING'),
  'thermal_film':('p0*(tau-1)*h/(1+p1*h)',['thermal-film coupling strength','positive film attenuation'],'UNCONSTRAINED')}
}
for choices in CATALOG.values():
    choices['null']=('0',[],'CONSTANT')

INSTRUCTIONS='''Choose exactly ONE construction_id from the supplied predefined graph
for the assigned hypothesis. Explain its scientific assumptions, source relevance,
limitations and falsifiable prediction. These are log-current contributions, not raw
rates. The program supplies the exact expression, variables, parameter signs,
derivatives and shape label. Do not output formula strings or parameter definitions.
If none is scientifically defensible, return proposals=[] and explain why_stop_adding_forms.
There is one request only: no mathematical repair or schema regeneration. Selecting a
valid mathematical construction does not establish its physical correctness.
'''

def schema(hid):
    result=compact.schema(hid)
    item=result['properties']['proposals']['items']
    for field in ('formula_py','formula_kind','parameters','conditions','expected_shape','shape_expectation'):
        item['properties'].pop(field);item['required'].remove(field)
    item['properties']['construction_id']={'type':'string','enum':list(CATALOG[hid])}
    item['required'].append('construction_id')
    result['properties']['proposals']['maxItems']=1
    return result

def prepare_payload(payload):
    result=copy.deepcopy(payload)
    hid=result['proposal_slot_contract']['assigned_hypothesis_id']
    result['instructions']=INSTRUCTIONS
    for field in ('repair_records','mathematical_repair_contract','operator_interface','executable_contract','operator_library'):
        result.pop(field,None)
    result['protocol'].pop('symbol_registry',None)
    graph={'version':VERSION,'hypothesis_id':hid,'response_scale':'log-current contribution',
           'coordinate_domains':result['protocol']['domains'],
           'reference':result['protocol']['reference'],
           'nodes':{k:{'expression':v[0],'positive_parameters':v[1],'trend':v[2]}
                    for k,v in CATALOG[hid].items()},
           'edges':[{'from':hid,'relation':'allows_construction','to':k} for k in CATALOG[hid]],
           'scientific_validity':'Requires scientific justification and subsequent evaluation'}
    result['predefined_math_graph']={**graph,'sha256':fingerprint(graph)}
    result['shared_symbol_knowledge']={'rules':['Executable symbols and parameter domains are owned by the predefined construction graph.'],
        'scientific_memory':result.get('shared_symbol_knowledge',{}).get('recent_failures',[])}
    return result

def compile_operator(raw):
    hid=raw['hypothesis_id'];key=raw['construction_id']
    formula,meanings,trend=CATALOG[hid][key]
    candidate=copy.deepcopy(raw)
    candidate.update(formula_py=formula,formula_kind='LOG_RESPONSE',conditions=[],
        parameters=[{'domain':'POSITIVE','meaning':m} for m in meanings],expected_shape=trend,
        shape_expectation='Construction graph trend: '+trend+'; fitted domain behavior must be evaluated.')
    if key=='null':
        candidate['question_links']=[{**link,'relation':'RIVAL_CONTROL'} for link in candidate['question_links']]
    result=compact.compile_operator(candidate)
    result['interface_receipt']['construction_graph']={'version':VERSION,'id':key,
        'definition_sha256':fingerprint(CATALOG[hid][key]),'formula_generated_by':'deterministic compiler'}
    return result
