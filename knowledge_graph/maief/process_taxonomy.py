"""Orthogonal classification and append-only revisions for mathematical knowledge."""
import ast
import copy
from .storage import fingerprint, save

PROCESSES={
 'T_ACTIVATED':('温度激活','H1'), 'T_LOCAL':('温度局部近似或竞争对照','H1'),
 'C_MONOTONE':('盐单调响应','H2'), 'C_COMPETING_PEAK':('盐促进与抑制竞争','H2'),
 'TC_PEAK_SHIFT':('温度改变盐峰位置','H4'),
 'H_MONOTONE':('液膜单调响应','H3'), 'H_COMPETING_PEAK':('液膜可用性与传输竞争峰','H3'),
 'CH_LENGTH':('盐改变液膜特征长度','H4'), 'CH_SLOPE':('盐改变液膜斜率或峰位','H4'),
 'CH_PEAK_SHAPE':('盐改变液膜峰幅度或曲率','H4'),
 'TH_COUPLING':('温度改变液膜响应','H4'), 'TCH_COUPLING':('温度与盐共同改变液膜响应','H4')}

def seal(graph):
    graph=copy.deepcopy(graph)
    graph.pop('sha256',None)
    graph['edges']=sorted(graph['edges'],key=lambda e:(e['source'],e['relation'],e['target']))
    for e in graph['edges']:
        if e['source'] not in graph['nodes'] or e['target'] not in graph['nodes']:
            raise ValueError('DANGLING_GRAPH_EDGE')
    graph['sha256']=fingerprint(graph)
    return graph

def attach_template(graph,tid,process_ids,role='phenomenological_closure'):
    n=graph['nodes'][tid]
    n['process_family_ids']=['process:'+p for p in process_ids]
    tree=ast.parse(n['formula_py'],mode='eval')
    names={x.id for x in ast.walk(tree) if isinstance(x,ast.Name)}
    axes=sorted(names & {'tau','c','h'})
    forms=set()
    for f,label in [('exp','exponential'),('log','logarithmic'),('log1p','logarithmic'),('tanh','hyperbolic'),('sqrt','radical')]:
        if f in names:forms.add(label)
    if any(isinstance(x,ast.Div) for x in ast.walk(tree)):forms.add('rational_expression')
    if any(isinstance(x,ast.Pow) and isinstance(x,ast.AST) for x in ast.walk(tree)):
        forms.add('power_expression')
    if not forms:forms.add('constant' if not axes else 'polynomial')
    tags={'form':sorted(forms),'shape':[n['expected_shape']],
          'variables':[('single:' if len(axes)==1 else 'multi:')+','.join(axes)] if axes else ['none'],
          'role':[role]}
    n['tags']=tags
    n['shape_claim_scope']='Declared conditional shape; peak need not be inside measured interval; numerical checks are not global proofs.'
    n.setdefault('validation_state','UNCHECKED')
    n.setdefault('evidence_state','NOT_EVALUATED')
    for pid in n['process_family_ids']:
        if pid not in graph['nodes']:raise ValueError('UNKNOWN_PROCESS_FAMILY')
        if n['hypothesis_id'] not in graph['nodes'][pid]['hypothesis_ids']:raise ValueError('PROCESS_HYPOTHESIS_MISMATCH')
        graph['edges'].append({'source':pid,'relation':'has_candidate_template','target':tid})
    for axis,values in tags.items():
        for value in values:
            tag='tag:'+axis+':'+value
            graph['nodes'].setdefault(tag,{'id':tag,'kind':'classification_tag','dimension':axis,'value':value})
            graph['edges'].append({'source':tid,'relation':'has_'+axis+'_tag','target':tag})

def upgrade(graph):
    graph=copy.deepcopy(graph)
    old={k for k,v in graph['nodes'].items() if v['kind']=='mathematical_family'}
    graph['nodes']={k:v for k,v in graph['nodes'].items() if k not in old}
    graph['edges']=[e for e in graph['edges'] if e['source'] not in old and e['target'] not in old]
    for key,(label,hid) in PROCESSES.items():
        id='process:'+key
        graph['nodes'][id]={'id':id,'kind':'process_family','label':label,'hypothesis_ids':[hid],
            'origin':'Historical V4 family audit taxonomy; categories only, not inherited fitted coefficients or scientific validation',
            'scientific_status':'PROVISIONAL_PROCESS_OR_RIVAL'}
        if key=='TC_PEAK_SHIFT':
            graph['nodes'][id]['implementation_constraint']='Current H4 execution contract requires h; a tau-c-only template needs an explicit slot-contract extension.'
        graph['edges'].append({'source':'hypothesis:'+hid,'relation':'investigates_process','target':id})
    for id,n in list(graph['nodes'].items()):
        if n['kind']!='executable_template':continue
        hid=n['hypothesis_id'];key=id.rsplit(':',1)[-1]
        n['legacy_category']=n.pop('family_id')
        if key=='null':process=[];role='null_control'
        elif hid=='H1':process=['T_ACTIVATED' if key=='activated' else 'T_LOCAL'];role='phenomenological_closure'
        elif hid=='H2':process=['C_COMPETING_PEAK' if key in ['competition','gaussian_log_response','bell_log1p','power_bell_log1p'] else 'C_MONOTONE'];role='phenomenological_closure'
        elif hid=='H3':process=['H_COMPETING_PEAK' if key in ['transport_peak','gaussian_log_response','bell_log1p','power_bell_log1p'] else 'H_MONOTONE'];role='phenomenological_closure'
        else:process=['TH_COUPLING' if key=='thermal_film' else 'CH_SLOPE'];role='phenomenological_closure'
        n['family_id']=process[0] if process else 'NULL_CONTROL'
        attach_template(graph,id,process,role)
    for key in PROCESSES:
        node=graph['nodes']['process:'+key]
        ids=[n['id'] for n in graph['nodes'].values() if n['kind']=='executable_template' and 'process:'+key in n['process_family_ids']]
        node['template_ids']=ids
        node['coverage_status']='INITIAL_TEMPLATES_AVAILABLE' if ids else 'TEMPLATE_GAP'
    graph.update(version='PROCESS_MULTITAG_KNOWLEDGE_GRAPH_2',revision=0,parent_sha256=None,
        revision_events=[{'kind':'INITIAL_TAXONOMY','detail':'12 process entry points; independent form/shape/variable/role tags; 8 legacy categories retained only as provenance.'}],
        state_policy={'validation_state':['UNCHECKED','MATH_CHECKED','QUARANTINED'],
                      'evidence_state':['NOT_EVALUATED','EVALUATED_IN_SCOPE'],
                      'rule':'Mathematical checks never imply empirical support; evaluation records do not imply a universal physical law.'})
    return seal(graph)

def register_candidate(graph,template,process_ids,scientific_reason):
    """Return a new graph; never mutate or replace the caller's snapshot."""
    if not scientific_reason.strip():raise ValueError('SCIENTIFIC_REASON_REQUIRED')
    if template['id'] in graph['nodes']:raise ValueError('IMMUTABLE_TEMPLATE_ID')
    from .math_knowledge import compile_template
    from .generation import verify_extended
    from .storage import read
    from .math_knowledge import ROOT
    work=copy.deepcopy(graph)
    t=copy.deepcopy(template);t.update(kind='executable_template',validation_state='UNCHECKED',evidence_state='NOT_EVALUATED')
    for stale in ('validation_summary','math_checks'):
        t.pop(stale,None)
    work['nodes'][t['id']]=t
    attach_template(work,t['id'],process_ids)
    work['nodes'][t['id']]['family_id']=process_ids[0] if process_ids else 'UNASSIGNED'
    try:
        checks=verify_extended(compile_template(work,t['id'],scientific_reason),read(ROOT/'inputs/module_protocol.json'),[])
    except (ValueError,KeyError,TypeError,SyntaxError) as exc:
        checks={'errors':[type(exc).__name__+': '+str(exc)]}
    work['nodes'][t['id']]['validation_state']='QUARANTINED' if checks['errors'] else 'MATH_CHECKED'
    work['nodes'][t['id']]['math_checks']=checks
    for pid in process_ids:
        work['nodes']['process:'+pid]['template_ids'].append(t['id'])
        work['nodes']['process:'+pid]['coverage_status']='CANDIDATES_REGISTERED_CHECK_STATUS_REQUIRED'
    work.update(parent_sha256=graph['sha256'],revision=graph['revision']+1)
    work['revision_events'].append({'kind':'REGISTER_CANDIDATE','template_id':t['id'],'reason':scientific_reason})
    return seal(work)

def snapshot(graph,directory):
    from pathlib import Path
    target=Path(directory)/(graph['sha256']+'.json')
    if target.exists():
        from .storage import read
        if read(target)!=graph:raise ValueError('SNAPSHOT_CONTENT_MISMATCH')
    else:save(target,graph)
    return target

def record_evaluation(graph,template_id,evidence_reference,metrics):
    """Attach supplied numerical evidence; never assert physical truth."""
    if not evidence_reference.strip() or not metrics:raise ValueError('EVALUATION_EVIDENCE_REQUIRED')
    if graph['nodes'][template_id].get('validation_state')!='MATH_CHECKED':
        raise ValueError('MATHEMATICAL_CHECK_REQUIRED')
    work=copy.deepcopy(graph)
    record={'template_id':template_id,'evidence_reference':evidence_reference,
            'metrics':copy.deepcopy(metrics),'verification':'Supplied by caller; not independently verified by graph'}
    id='evaluation:'+fingerprint(record)
    if id in work['nodes']:return work
    work['nodes'][id]={'id':id,'kind':'numerical_evaluation',**record}
    work['nodes'][template_id]['evidence_state']='EVALUATED_IN_SCOPE'
    work['edges'].append({'source':template_id,'relation':'has_numerical_evaluation','target':id})
    work.update(parent_sha256=graph['sha256'],revision=graph['revision']+1)
    work['revision_events'].append({'kind':'RECORD_EVALUATION','evaluation_id':id})
    return seal(work)

def register_process(graph,key,label,hypothesis_id,scientific_reason):
    id='process:'+key
    if id in graph['nodes']:raise ValueError('IMMUTABLE_PROCESS_ID')
    if 'hypothesis:'+hypothesis_id not in graph['nodes'] or not label.strip() or not scientific_reason.strip():
        raise ValueError('PROCESS_CONTRACT_REQUIRED')
    work=copy.deepcopy(graph)
    work['nodes'][id]={'id':id,'kind':'process_family','label':label,'hypothesis_ids':[hypothesis_id],
        'origin':scientific_reason,'scientific_status':'PROVISIONAL_PROCESS_OR_RIVAL',
        'template_ids':[],'coverage_status':'TEMPLATE_GAP'}
    work['edges'].append({'source':'hypothesis:'+hypothesis_id,'relation':'investigates_process','target':id})
    work.update(parent_sha256=graph['sha256'],revision=graph['revision']+1)
    work['revision_events'].append({'kind':'REGISTER_PROCESS','process_id':id,'reason':scientific_reason})
    return seal(work)
