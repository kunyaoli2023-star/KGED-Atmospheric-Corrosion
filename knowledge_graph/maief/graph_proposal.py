"""LLM proposes scientific selections/recipes; the saved graph owns all algebra."""
import copy
from . import one_shot
from .math_knowledge import ROOT, compile_template, compile_combination
from .process_taxonomy import seal
from .storage import read
from .storage import fingerprint

GRAPH=read(ROOT/'math_graph.json' if (ROOT/'math_graph.json').exists() else ROOT/'knowledge/math_graph.json')
if GRAPH!=seal(GRAPH):raise ValueError('GRAPH_HASH_MISMATCH')
INSTRUCTIONS='''Use the provided saved mathematical knowledge graph as the authoritative
construction library. For the assigned hypothesis propose several scientifically
justified competing forms in ONE response (up to six). Do not output formulas,
parameter lists, derivatives or low-level nodes. Choose process_id and template_id;
use SELECT with right_template_id='', or SUM_LOG_RESPONSES / DIFFERENCE_LOG_RESPONSES
with a distinct right template. Both templates must belong to the assigned hypothesis.
All expressions are centered contributions to log current, not raw current factors.
For SELECT the process must be listed in the template process_family_ids. Null
templates require process_id=CONTROL and SELECT. Compositions may test another
process of the same hypothesis; explain that new process assumption explicitly.
Explain source relevance, scientific assumptions, mathematical differences and limits.
question_links must link to the assigned H1-H4 question; a null is a RIVAL_CONTROL,
not an IMPLEMENTATION. contrast_links refer ONLY to IDs in new_subhypotheses,
NEVER to another proposal ID. If no registered subhypotheses are supplied,
contrast_links MUST be []. Describe candidate comparisons in mathematical_distinction.
Report missing process routes in coverage_not_claimed. A graph process without an
executable template is a gap, never permission to invent an ID or equation.
Aim for distinct viable forms and a justified composition when useful, not arbitrary
combinations to meet a quota. No mathematical repair or request retry. Scientific
support is not implied by graph membership or numerical executability.
'''

def templates(hid):
    return {k:v for k,v in GRAPH['nodes'].items() if v['kind']=='executable_template'
            and v['hypothesis_id']==hid and v['validation_state']=='MATH_CHECKED'}

def processes(hid):
    return {k:v for k,v in GRAPH['nodes'].items() if v['kind']=='process_family' and hid in v['hypothesis_ids']}

def recipe_catalog(hid):
    ids=sorted(templates(hid));nonzero=[i for i in ids if templates(hid)[i]['formula_py']!='0']
    rows=[{'template_id':i,'right_template_id':'','operation':'SELECT'} for i in ids]
    rows += [{'template_id':a,'right_template_id':b,'operation':op}
             for op in ('SUM_LOG_RESPONSES','DIFFERENCE_LOG_RESPONSES')
             for a in nonzero for b in nonzero if a!=b and (op!='SUM_LOG_RESPONSES' or a<b)]
    return {'recipe:'+hid+':'+fingerprint(r)[:16]:r for r in rows}


def novel_recipes(hid,pool):
    used=set()
    for m in pool:
        p=m['proposal']
        if p['hypothesis_id']!=hid:continue
        intent=p.get('interface_receipt',{}).get('graph_intent')
        if intent:
            r={k:intent[k] for k in ('template_id','right_template_id','operation')}
            if r['operation']=='SUM_LOG_RESPONSES':r['template_id'],r['right_template_id']=sorted((r['template_id'],r['right_template_id']))
            used.add(fingerprint(r))
    return {k:v for k,v in recipe_catalog(hid).items() if fingerprint(v) not in used}


def schema(hid, contrast_ids=(), recipes=None):
    result=one_shot.schema(hid);item=result['properties']['proposals']['items']
    item['properties'].pop('construction_id');item['required'].remove('construction_id')
    fields={'process_id':list(processes(hid))+['CONTROL'],
            'template_id':list(templates(hid)),
            'right_template_id':['']+list(templates(hid)),
            'operation':['SELECT','SUM_LOG_RESPONSES','DIFFERENCE_LOG_RESPONSES']}
    for name,values in fields.items():
        item['properties'][name]={'type':'string','enum':values};item['required'].append(name)
    result['properties']['proposals']['maxItems']=6
    links=item['properties']['contrast_links']
    if contrast_ids:
        links['items']['properties']['contrast_id']={'type':'string','enum':list(contrast_ids)}
    else:
        links['maxItems']=0
    if recipes is not None:
        for name in ('template_id','right_template_id','operation'):
            item['properties'].pop(name);item['required'].remove(name)
        item['properties']['recipe_id']={'type':'string','enum':list(recipes) or ['NO_NOVEL_RECIPE_AVAILABLE']}
        item['required'].append('recipe_id')
        if not recipes:result['properties']['proposals']['maxItems']=0
    return result

def prepare_payload(payload):
    hid=payload['proposal_slot_contract']['assigned_hypothesis_id']
    result={k:copy.deepcopy(payload[k]) for k in ['question','evidence','hypothesis_state','proposal_slot_contract',
        'construction_plan','construction_focus','new_subhypotheses','feedback','existing_modules','shared_symbol_knowledge'] if k in payload}
    result['instructions']=INSTRUCTIONS
    result['mathematical_graph']={'version':GRAPH['version'],'sha256':GRAPH['sha256'],
        'response_contract':GRAPH['response_contract'],'processes':processes(hid),
        'templates':templates(hid),'coordinates':{k:v for k,v in GRAPH['nodes'].items() if v['kind']=='coordinate'},
        'construction_rules':{k:v for k,v in GRAPH['nodes'].items() if v['kind']=='construction_rule'}}
    return result

def compile_operator(raw):
    if 'recipe_id' in raw:
        choices=recipe_catalog(raw['hypothesis_id'])
        if raw['recipe_id'] not in choices:raise ValueError('UNKNOWN_GRAPH_RECIPE')
        raw={**raw,**choices[raw['recipe_id']]}
    hid=raw['hypothesis_id'];left=raw['template_id'];right=raw['right_template_id'];op=raw['operation'];pid=raw['process_id']
    if left not in templates(hid):raise ValueError('INVALID_TEMPLATE_SCOPE')
    if pid!='CONTROL' and pid not in processes(hid):raise ValueError('INVALID_PROCESS_SCOPE')
    node=templates(hid)[left]
    if op=='SELECT':
        if right:raise ValueError('SELECT_REQUIRES_EMPTY_RIGHT')
        if pid=='CONTROL':
            if node['formula_py']!='0':raise ValueError('CONTROL_REQUIRES_NULL')
        elif pid not in node['process_family_ids']:raise ValueError('PROCESS_TEMPLATE_MISMATCH')
        p=compile_template(GRAPH,left,raw['scientific_argument'])
    else:
        if right not in templates(hid) or pid=='CONTROL' or node['formula_py']=='0' or templates(hid)[right]['formula_py']=='0':
            raise ValueError('INVALID_COMBINATION_SCOPE')
        p=compile_combination(GRAPH,left,right,op,raw['scientific_argument'])
    for key in ['proposal_id','source_links','evidence_gap','construction_family_id','mathematical_distinction','question_links','contrast_links']:
        p[key]=copy.deepcopy(raw[key])
    p['family']=pid
    p['interface_receipt']['graph_intent']={k:raw[k] for k in ['process_id','template_id','right_template_id','operation']}
    return p
