"""Build and validate the standalone mathematical graph without any LLM call."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from maief.math_knowledge import build_graph, compile_template, compile_combination, ROOT
from maief.generation import verify_extended
from maief.numerics import Module
from maief.storage import read, save, fingerprint
from maief.process_taxonomy import seal, snapshot

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'knowledge');args=parser.parse_args()
    graph=build_graph();protocol=read(ROOT/'inputs/module_protocol.json');results=[]
    assert graph==build_graph(),'Nondeterministic graph'
    assert all(e['source'] in graph['nodes'] and e['target'] in graph['nodes'] for e in graph['edges'])
    for id,node in graph['nodes'].items():
        if node['kind']!='executable_template':continue
        p=compile_template(graph,id,'Graph construction test; not a scientific claim.')
        checks=verify_extended(p,protocol,[])
        numeric_error=None
        try:
            module=Module({'module_id':id,'proposal':p})
            values={v:np.linspace(*protocol['domains'][v],9) for v in ['tau','c','h']}
            if module.n:
                y,jac=module.response(values,np.zeros(module.n),jac=True)
                assert np.isfinite(y).all() and np.isfinite(jac).all()
            else:assert np.isfinite(module.response(values,np.zeros(0))).all()
        except Exception as exc:numeric_error=type(exc).__name__+': '+str(exc)
        results.append({'template_id':id,'errors':checks['errors'],'warnings':checks['warnings'],
                        'numerical_error':numeric_error,'dedup_key':checks.get('dedup_key')})
    composed=compile_combination(graph,'template:H2:saturation','template:H2:logarithmic',
        'DIFFERENCE_LOG_RESPONSES','Test of distinct positive parameter slices.')
    combination_check=verify_extended(composed,protocol,[])
    summed=compile_combination(graph,'template:H2:saturation','template:H2:logarithmic',
        'SUM_LOG_RESPONSES','Test of distinct positive parameter slices.')
    sum_check=verify_extended(summed,protocol,[])
    for result in results:
        n=graph['nodes'][result['template_id']]
        n['validation_state']='MATH_CHECKED' if not result['errors'] and not result['numerical_error'] else 'QUARANTINED'
        n['validation_summary']=result
    graph=seal(graph)
    counts=dict(Counter(n['kind'] for n in graph['nodes'].values()))
    report={'graph_sha256':graph['sha256'],'node_counts':counts,'edges':len(graph['edges']),
            'templates_checked':len(results),'passed':sum(not r['errors'] and not r['numerical_error'] for r in results),
            'checks':results,'combination_check':{'expression':composed['expression'],'errors':combination_check['errors']},
            'sum_combination_check':{'expression':summed['expression'],'errors':sum_check['errors']},
            'scope':'All templates checked locally at finite probes; numerical engine response/Jacobian checked at unit positive parameters. No LLM, parameter fitting or scientific validation.'}
    previous=args.out/'math_graph.json'
    if previous.exists():snapshot(read(previous),args.out/'snapshots')
    snapshot(graph,args.out/'snapshots')
    save(previous,graph);save(args.out/'validation.json',report)
    print({k:v for k,v in report.items() if k not in ['checks']})
    if report['passed']!=len(results) or combination_check['errors'] or sum_check['errors']:raise SystemExit(1)

if __name__=='__main__':main()
