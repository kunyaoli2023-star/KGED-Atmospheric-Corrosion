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
    # A small self-contained view for inspection; no external JS dependencies.
    import html
    rows=[]
    for id,n in graph['nodes'].items():
        if n['kind']=='executable_template':
            rows.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [id,n['hypothesis_id'],n['process_family_ids'],n['tags'],n['formula_py'],n['validation_state']])+'</tr>')
    page='''<!doctype html><meta charset="utf-8"><title>V6 数学知识图谱</title>
<style>body{font:16px system-ui;margin:32px;color:#172b4d}table{border-collapse:collapse;width:100%}td,th{padding:10px;border:1px solid #ccd5df;text-align:left}tr:nth-child(even){background:#f2f6fa}input{padding:10px;width:60%}code{background:#eef}</style>
<h1>V6 过程主族与多维数学标签知识图谱</h1><p>原始量 → 无量纲符号 → H1–H4 → 12 个过程主族 → 可执行模板 ↔ 形式 / 形状 / 变量 / 角色标签</p>
<p>这是一份数学构造库，不代表所列机理已被实验证实。参数由程序生成，LLM 不需要填写底层节点。</p>'''
    page+='<h2>过程主族与覆盖状态</h2><table><tr><th>主族</th><th>说明</th><th>假设</th><th>模板数</th><th>状态</th></tr>'
    for n in graph['nodes'].values():
        if n['kind']=='process_family':
            page+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [n['id'],n['label'],n['hypothesis_ids'],len(n['template_ids']),n['coverage_status']])+'</tr>'
    page+='</table><h2>可执行模板及多维标签</h2><p>'+html.escape(str(counts))+'</p><input id="q" placeholder="筛选 H2、过程族或公式…"><table><thead><tr><th>模板</th><th>假设</th><th>过程主族</th><th>独立标签</th><th>对数响应表达式</th><th>数学状态</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>'
    page+="<script>document.getElementById('q').addEventListener('input',e=>{for(const r of document.querySelectorAll('tbody tr'))r.hidden=!r.textContent.toLowerCase().includes(e.target.value.toLowerCase())});</script>"
    args.out.mkdir(parents=True,exist_ok=True);(args.out/'index.html').write_text(page,encoding='utf-8')
    print({k:v for k,v in report.items() if k not in ['checks']})
    if report['passed']!=len(results) or combination_check['errors'] or sum_check['errors']:raise SystemExit(1)

if __name__=='__main__':main()
