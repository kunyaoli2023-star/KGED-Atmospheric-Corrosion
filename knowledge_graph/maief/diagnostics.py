"""Code-only checkpoints and post-run comparison; never search feedback."""
from datetime import datetime, timezone
import html
import json
from pathlib import Path
from .storage import read, save, sha


def cycle_reasons(row):
    reasons=[]
    if row.get('material_archive_progress',row.get('archive_progress',False)):
        reasons.append('MATERIAL_PARETO_PROGRESS')
    if row.get('hypothesis_material_change',row.get('hypothesis_changed',False)):
        reasons.append('SCIENTIFIC_STATE_OR_NOVELTY_CHANGE')
    if row.get('post_selection_review_pending',False):
        reasons.append('NEW_SELECTED_EQUATION_AWAITS_SCIENTIFIC_REVIEW')
    if row.get('uncertain_parent_updates',0):
        reasons.append('UNRESOLVED_PARENT_STATE_AUDIT')
    if row.get('uncertain_hypothesis_contrasts',0):
        reasons.append('UNRESOLVED_NEW_CONTRAST_AUDIT')
    if not row.get('evaluation_complete_under_solver_policy',False):
        reasons.append('DECLARED_BATCH_INCOMPLETE')
    if row.get('pending_candidates',0):
        reasons.append('PENDING_NUMERICAL_CANDIDATES')
    if not reasons:
        reasons.append('STOP_CRITERION_MET' if row.get('stop') else 'STABLE_CYCLE_ACCUMULATING')
    return reasons


def diagnostic(state, status, stop_reason):
    rows=[]
    for r in state['rounds']:
        score=r.get('default_score',{})
        rows.append({'cycle':r['round']+1,'best_cv':r.get('best_cv'),
            'default_cv':score.get('cv_log_rmse'),'parameters':score.get('parameter_count'),
            'ast':score.get('canonical_ast_nodes'),'stable_cycles':r.get('stable_rounds'),
            'new_modules':r.get('new_admitted_modules',0),
            'new_contrasts':r.get('new_subhypotheses',0),
            'uncertain_parents':r.get('uncertain_parent_updates',0),
            'uncertain_contrasts':r.get('uncertain_hypothesis_contrasts',0),
            'numerical_exclusions':r.get('numerical_exclusions',0),
            'continuation_reasons':cycle_reasons(r)})
    return {'time_utc':datetime.now(timezone.utc).isoformat(),'status':status,
        'stop_reason':stop_reason,'completed_cycles':state['round'],
        'stable_cycles':state['stale'],'trajectory':rows,
        'current_parent_states':state['hypotheses'],
        'registered_children':len(state['new_hypotheses']),
        'pending_parent_updates':state.get('pending_parent_hypothesis_updates',[]),
        'pending_contrasts':state.get('pending_hypothesis_contrasts',[]),
        'checkpoint_is_not_a_stop_rule':True,'llm_calls_by_this_diagnostic':0,
        'historical_results_used_as_search_feedback':False,
        'global_convergence_claimed':False}


def write_diagnostic(root, state, name, status='RUNNING', reason='CONTINUE'):
    result=diagnostic(state,status,reason)
    folder=Path(root)/'diagnostics'
    save(folder/(name+'.json'),result)
    rows=''.join('<tr>'+''.join('<td>'+html.escape(str(r[key]))+'</td>' for key in
        ('cycle','default_cv','best_cv','parameters','ast','stable_cycles','new_contrasts',
         'uncertain_parents','uncertain_contrasts','continuation_reasons'))+'</tr>'
        for r in result['trajectory'])
    body=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>自动运行诊断</title>
<style>body{{font:15px/1.7 system-ui;max-width:1350px;margin:30px auto;padding:20px;color:#203541}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border:1px solid #d8e2e8}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style>
<h1>自动运行诊断：已完成 {state['round']} 轮</h1>
<p>状态：{html.escape(status)}；{html.escape(reason)}。稳定周期：{state['stale']}。</p>
<p>第9轮只是诊断检查点，不强制停止、不改变科学门槛。每轮先完成数值竞争、科学反馈和新形式评价，再判断停止。</p>
<table><tr><th>轮次</th><th>默认CV</th><th>最佳CV</th><th>参数</th><th>AST</th><th>稳定周期</th><th>新对比</th><th>父状态待定</th><th>新对比待定</th><th>继续或停止原因</th></tr>{rows}</table>
<p>上表CV是反复使用的开发分组验证，不是独立测试集。数值排除保留，逐轮稳定不等于穷举或全局收敛。</p>
<details><summary>当前假设与完整诊断</summary><pre>{html.escape(json.dumps(result,ensure_ascii=False,indent=2))}</pre></details></html>'''
    (folder/(name+'.html')).write_text(body,encoding='utf-8')
    return result


def run_summary(root):
    root=Path(root).resolve()
    summary=read(root/'summary.json')
    retained=root/'retained_equation.json'
    result={'run':str(root),'status':summary['status'],'stop_reason':summary['stop_reason'],
        'completed_cycles':summary['completed_rounds'],
        'historical_continuation':(root/'CONTINUATION_RECEIPT.json').exists(),
        'model':summary.get('real_model'),'modules':summary.get('modules')}
    if retained.exists():
        r=read(retained)
        result.update(score=r['score'],intercept=r['intercept'],
            physical_modules=r['physical_modules'],full_predictions=r['full_predictions'],
            equation_sha256=sha(retained))
    field=root/'engineering_results.json'
    if field.exists():
        result['default_engineering']=[r['scores'] for r in read(field)['representatives']
            if r.get('default') and r.get('completed')]
    return result


def compare_after_run(root, reference, out):
    """Reference is opened ONLY after the discovery call has returned/failed."""
    current,old=run_summary(root),run_summary(reference)
    checks={name:sha(Path(root)/'inputs'/name)==sha(Path(reference)/'inputs'/name)
        for name in ('data.json','folds.json','module_protocol.json','hypotheses.json')}
    result={'scope':'Post-run comparison only; reference was not supplied to search.',
        'current':current,'reference':old,'matching_inputs':checks,
        'old_winner_imported_into_this_search':False}
    if 'score' in current and 'score' in old:
        result['differences']={name:current['score'][name]-old['score'][name]
            for name in ('cv_log_rmse','parameter_count','canonical_ast_nodes')}
        if checks['data.json']:
            result['max_training_log_prediction_difference']=max(abs(a-b) for a,b in
                zip(current['full_predictions'],old['full_predictions']))
    result['provenance_interpretation']=('Reference imported a previous pool/archive; this fresh run did not. Same stopping rule and model do not make the initial search state identical.'
        if old['historical_continuation'] else 'Check model outputs and candidate coverage; an identical LLM name does not ensure identical generated forms.')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    save(out/'comparison.json',result)
    rows=''.join('<tr><td>'+html.escape(name)+'</td><td>'+html.escape(str(current.get('score',{}).get(name)))+
        '</td><td>'+html.escape(str(old.get('score',{}).get(name)))+'</td></tr>' for name in
        ('cv_log_rmse','parameter_count','canonical_ast_nodes','response_jacobian_condition'))
    text=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>整程结束后的比较</title>
<style>body{{font:16px/1.8 system-ui;max-width:1100px;margin:35px auto;padding:20px}}td,th{{padding:10px;border-bottom:1px solid #ccc}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style>
<h1>整程结束后的程序化比较</h1><p>本次 {current['completed_cycles']} 轮，{html.escape(current['status'])}；参考记录 {old['completed_cycles']} 轮。</p>
<table><tr><th>指标</th><th>本次输出</th><th>参考输出</th></tr>{rows}</table>
<p>{html.escape(result['provenance_interpretation'])}</p>
<p>结构差异不等于失败；应分别判断误差、参数数目、AST、数值质量及停止依据。参考方程和工程目标误差均未用于本次选式。</p>
<details><summary>方程、工程结果与完整比较数据</summary><pre>{html.escape(json.dumps(result,ensure_ascii=False,indent=2))}</pre></details></html>'''
    (out/'report.html').write_text(text,encoding='utf-8')
    return result
