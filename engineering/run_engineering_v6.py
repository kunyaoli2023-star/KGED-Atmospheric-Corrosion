"""Compare frozen V6 laboratory laws using the local historical field protocol."""
import argparse
import shutil
import sys
from pathlib import Path
import numpy as np
from maief.storage import read, save, sha
from maief.numerics import Module
from maief.symbols import REFERENCE
from maief.engineering_integrator import CASES, Integrator, metrics
from maief import engineering_state as state

ROOT=Path(__file__).resolve().parent


class FrozenLaw:
    def __init__(self, result, pool):
        self.result=result
        self.modules=[]
        for p in result['physical_modules']:
            m=Module(pool[p['module_id']])
            if m.p['expression']!=p['expression']: raise ValueError('EXPRESSION_MISMATCH')
            z=np.asarray([np.log(p['parameters'][q['name']]) if q['domain']=='POSITIVE' else p['parameters'][q['name']] for q in m.params])
            self.modules.append((m,z))

    def predict_log(self,T,C,H):
        T,C,H=np.broadcast_arrays(np.atleast_1d(np.asarray(T,float)),np.atleast_1d(np.asarray(C,float)),np.atleast_1d(np.asarray(H,float)))
        shape=T.shape
        if not all(np.isfinite(x).all() for x in (T,C,H)) or np.any(T<=-273.15) or np.any(C<0) or np.any(H<=0):
            raise ValueError('INVALID_ENGINEERING_INPUT')
        x={'tau':(T.ravel()+273.15)/298.15,'c':C.ravel(),'h':H.ravel()/200.}
        y=np.full(T.size,self.result['intercept'],float)
        for m,z in self.modules:
            if np.any(m.constraints(x,z)<-1e-8): raise ValueError('FORMULA_CONDITION_FAILED')
            y+=m.response(x,z)
        if not np.isfinite(y).all(): raise ValueError('NONFINITE_LOG_CURRENT')
        return y.reshape(shape)


def run(source,out):
    source,out=source.resolve(),out.resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError('Use a new output directory')
    field=ROOT/'inputs/engineering'
    access={'scoring':False,'blocked_target_reads':0}
    forbidden={str((field/f'{c}_labels.json').resolve()).casefold() for c in CASES[1:]}
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],(str,Path)) and not access['scoring']:
            if str(Path(args[0]).resolve()).casefold() in forbidden:
                access['blocked_target_reads']+=1
                raise RuntimeError('TARGET_LABEL_ACCESS_BEFORE_FREEZE')
    sys.addaudithook(guard)
    front=read(source/'pareto_frontier.json')
    chosen={'parsimonious':read(source/'one_se_representative.json'),
            'best_accuracy':min(front,key=lambda r:(r['score']['cv_log_rmse'],r['score']['parameter_count'],r['equation_key']))}
    if any(not r['admissible'] or r['equation_key'] not in {f['equation_key'] for f in front} for r in chosen.values()):
        raise ValueError('INVALID_SELECTION')
    pool={m['module_id']:m for m in read(source/'checkpoint.json')['pool']}
    lab=read(source/'inputs/data.json')
    if read(source/'inputs/symbol_registry.json')!=read(ROOT/'inputs/symbol_registry.json'):
        raise ValueError('SYMBOL_CONTRACT_MISMATCH')
    mapping=read(field/'mapping.json')
    provenance=read(field/'IMPORT_PROVENANCE.json')
    data={}
    out.mkdir(parents=True)
    snapshots=out/'inputs'; snapshots.mkdir()
    for c in CASES:
        p=field/f'{c}_environment.npz'
        if sha(p)!=provenance['input_sha256'][p.name]: raise ValueError('ENVIRONMENT_CHANGED')
        with np.load(p) as z: data[c]={k:z[k].copy() for k in z.files}
        shutil.copy2(p,snapshots/p.name)
    for name in ('mapping.json','linhai_labels.json'):
        if sha(field/name)!=provenance['input_sha256'][name]: raise ValueError('SOURCE_PROTOCOL_CHANGED')
        shutil.copy2(field/name,snapshots/name)
    source_labels=read(field/'linhai_labels.json')
    months=np.asarray(source_labels['months'],float)
    observed=np.asarray(source_labels['observed_depth_um'],float)
    np.testing.assert_array_equal(months,data['linhai']['months'])
    frozen={'status':'FROZEN_BEFORE_TARGET_LABEL_SCORING','source_run':str(source),
        'selection_rule':'Saved one-SE parsimonious representative and minimum laboratory CV; no field scores used.',
        'source_input_sha256':{n:sha(source/n) for n in ('pareto_frontier.json','one_se_representative.json','checkpoint.json','inputs/data.json','inputs/symbol_registry.json')},
        'source_calibration_labels':len(observed),'target_calibration_labels':0,
        'mapping':mapping,'reference_coordinates':REFERENCE,'models':{},'domain_diagnostics':{}}
    for c,d in data.items():
        q=np.clip((d['RH_pct']-55.)/45.,0,1)**.3
        salt=mapping['Cbg']+(mapping['Ccap']-mapping['Cbg'])*d['chloride_deposition']/(d['chloride_deposition']+20000.)
        film=np.clip(1000*q,50,1000)
        outside=(d['T_C']<0)|(d['T_C']>50)
        frozen['domain_diagnostics'][c]={'temperature_range_C':[float(d['T_C'].min()),float(d['T_C'].max())],
            'effective_salt_range_wt_percent':[float(salt.min()),float(salt.max())],
            'effective_film_range_um':[float(film.min()),float(film.max())],
            'outside_lab_temperature_duration_fraction':float(d['duration_days'][outside].sum()/d['duration_days'].sum())}
    for alias,result in chosen.items():
        law=FrozenLaw(result,pool)
        reconstructed=law.predict_log(lab['T_C'],lab['C_wt'],lab['h_um'])
        delta=float(np.max(np.abs(reconstructed-np.asarray(result['full_predictions']))))
        np.testing.assert_allclose(reconstructed,result['full_predictions'],atol=2e-11,rtol=0)
        model=Integrator(data,law,mapping)
        fitted=state.fit(model,mapping['fixed_state_structure'],observed,months,len(observed))
        if not fitted['eligible']: raise ValueError('SOURCE_FIT_INELIGIBLE_'+alias)
        predictions={};checks=[]
        for c in CASES:
            b16=model.basis(c,mapping['fixed_state_structure'])
            b32=model.basis(c,mapping['fixed_state_structure'],32)
            np.testing.assert_allclose(b16,b32,atol=2e-9,rtol=2e-12)
            pred=b16@np.asarray(fitted['coefficients'])
            if not np.isfinite(pred).all() or np.any(pred<0) or np.any(np.diff(pred)<-1e-9): raise ValueError('INVALID_DEPTH')
            predictions[c]={'months':data[c]['months'].tolist(),'predicted_depth_um':pred.tolist()}
            checks.append({'case':c,'basis_16_vs_32_max_um':float(np.max(np.abs(b16-b32)))})
        save(out/f'{alias}_equation.json',result)
        frozen['models'][alias]={'equation_key':result['equation_key'],'lab_score':result['score'],
            'lab_reconstruction_max_log_difference':delta,'source_fit':fitted,
            'quadrature_checks':checks,'predictions':predictions}
    frozen['target_reads_before_freeze']=access['blocked_target_reads']
    frozen['input_sha256']={p.name:sha(p) for p in snapshots.iterdir()}
    frozen['implementation_sha256']={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'maief/engineering_integrator.py',ROOT/'maief/engineering_state.py',ROOT/'maief/numerics.py',ROOT/'maief/symbols.py',ROOT/'maief/construction.py')}
    save(out/'PREDICTIONS_FROZEN.json',frozen)
    frozen_sha=sha(out/'PREDICTIONS_FROZEN.json')
    access['scoring']=True
    labels={}; scores=[]
    for c in CASES:
        p=field/f'{c}_labels.json'
        if sha(p)!=provenance['input_sha256'][p.name]: raise ValueError('LABEL_SNAPSHOT_CHANGED')
        labels[c]=read(p)
        shutil.copy2(p,snapshots/p.name)
        for alias,model in frozen['models'].items():
            pred=model['predictions'][c]
            np.testing.assert_array_equal(labels[c]['months'],pred['months'])
            y=np.asarray(labels[c]['observed_depth_um']); yp=np.asarray(pred['predicted_depth_um'])
            scores.append({'model':alias,'case':c,'role':'source_calibration' if c=='linhai' else 'retrospective_target_prediction',
                **metrics(y,yp),'depth_rmse_um':float(np.sqrt(np.mean((yp-y)**2)))})
    if sha(out/'PREDICTIONS_FROZEN.json')!=frozen_sha: raise ValueError('FORECAST_CHANGED_DURING_SCORING')
    save(out/'scores.json',scores)
    save(out/'SCORING_RECEIPT.json',{'frozen_prediction_sha256':frozen_sha,'target_calibration_labels':0,
        'input_sha256':{p.name:sha(p) for p in snapshots.iterdir()},'scope':'Retrospective engineering comparison; sites previously used in project development.'})
    code=out/'implementation_snapshot';code.mkdir()
    for p in (Path(__file__),ROOT/'maief/engineering_integrator.py',ROOT/'maief/engineering_state.py'):
        shutil.copy2(p,code/p.name)
    render(out,chosen,frozen,scores,labels)
    print(__import__('json').dumps({'status':'COMPLETE','out':str(out),'scores':scores},ensure_ascii=False,indent=2))


def render(out,chosen,frozen,scores,labels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for ax,c in zip(axes.flat,CASES):
        ax.plot(labels[c]['months'],labels[c]['observed_depth_um'],'ko',label='Observed',ms=4)
        for alias,color,label in [('parsimonious','#2474b5','5-parameter parsimonious'),('best_accuracy','#dc7633','8-parameter best lab CV')]:
            p=frozen['models'][alias]['predictions'][c]
            ax.plot(p['months'],p['predicted_depth_um'],color=color,label=label,lw=1.8)
        ax.set(title=c.capitalize()+(' (source calibration)' if c=='linhai' else ' (target prediction)'),xlabel='Exposure time (months)',ylabel='Corrosion depth (µm)')
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.savefig(out/'ENGINEERING_PREDICTIONS.png',dpi=200)
    fig.savefig(out/'ENGINEERING_PREDICTIONS.pdf')
    plt.close(fig)
    lines=['# V6 最终两方程：工程预测比较','','## 选模与方程','','本次使用最终实验室前沿的 one-SE 简约代表（5 参数）和最低 CV 方程（8 参数）。简约代表不等于整个前沿的绝对最少参数模型；更简单的省略控制未被选作本次工程模型。工程分数不反馈到实验室选模。',
        '',r'变量：`tau=(T_C+273.15)/298.15`，`c=C_wt`（wt% 数值，例如 3.5），`h=h_um/200`。使用自然对数。',
        '',r'精确中心化方程：`ln(i) = b + Σ_j [f_j(tau,c,h) - f_j(1,3.5,1)]`。完整精度系数保存在对应 equation.json；不要省略参考项。','']
    for alias,result in chosen.items():
        lines += ['### '+alias,'',f"- equation_key: `{result['equation_key']}`",f"- 截距 b: `{result['intercept']:.16g}`",f"- CV log-RMSE: {result['score']['cv_log_rmse']:.10f}",f"- 实验室全数据 log-R²: {result['score']['train_log_r2']:.10f}",'','| 模块 | f_j 表达式 | 参数（模块局部） |','|---|---|---|']
        for m in result['physical_modules']:
            lines.append(f"| {m['hypothesis']} | `{m['expression']}` | "+', '.join(f"{k}={v:.16g}" for k,v in m['parameters'].items())+' |')
    lines += ['','## 工程流程','','沿用历史固定天气映射、剂量时钟和状态积分。只使用临海的 '+str(frozen['source_calibration_labels'])+' 个深度观测，分别标定两模型各 4 个工程状态幅度系数；实验室方程系数固定。目标站点标定标签为 0。两个模型预测先全部冻结，再读取金海、试验池和 KSC 标签评分。',
        '', '映射：`q=clip((RH-55)/45,0,1)^0.3`；`C_eff=Cbg+(Ccap-Cbg)*D/(D+20000)`；`H_eff=clip(1000*q,50,1000)`。固定映射参数及状态结构见 PREDICTIONS_FROZEN.json。工程累积腐蚀深度还依赖湿润时间和状态积分，不是把瞬时电流直接当作深度。',
        '', '工作电流单位沿用 A/cm² 与 Fe(II) Faraday 转换约定；原始仪器单位未在此重新确立。实验室的 5/8 参数之外，各有 4 个源站工程标定系数。',
        '', '## 工程误差','','| 模型 | 站点 | 角色 | 增量 R² | 终点实测 µm | 终点预测 µm | 终点误差 % | 深度 RMSE µm |','|---|---|---|---:|---:|---:|---:|---:|']
    for s in scores:
        r2='NA' if s['increment_r2'] is None else f"{s['increment_r2']:.4f}"
        lines.append(f"| {s['model']} | {s['case']} | {s['role']} | {r2} | {s['actual_final_depth_um']:.4f} | {s['predicted_final_depth_um']:.4f} | {s['endpoint_error_pct']:+.2f} | {s['depth_rmse_um']:.4f} |")
    lines += ['', '增量 R² 沿用旧协议：先减去首个观测时点的各自深度，再对后续增量计算；它不等同于绝对深度 R²。负值表示该增量评价较差。终点误差为相对实测的有符号百分误差。',
        '', '## 校验与边界','','- 两个方程逐点复原实验室存档预测（绝对容差 2e-11）。','- 16/32 点积分一致性、深度非负与累积单调性检查通过。','- 临海标定解的秩和条件数准入通过；目标标签未用于标定或选式。','- 本次为历史站点上的回顾性迁移比较，不是新盲测。有效盐度、液膜和湿润度为历史经验映射，不是实测膜厚与盐库存。','- 天气重复与 KSC 月度驱动限制沿用原协议；超出实验室温度范围的暴露比例见冻结记录。','- 所有工程运行代码及输入均位于 V6，本次预测没有运行时 V3/V4 导入。', '', '![工程预测](ENGINEERING_PREDICTIONS.png)','']
    (out/'ENGINEERING_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    run(args.run,args.out)
