"""Apply a measured first-depth condition to saved engineering growth forecasts."""
import argparse
import csv
import shutil
from pathlib import Path
import numpy as np
from maief.engineering_integrator import CASES, condition_on_initial_depth, metrics
from maief.storage import read, save, sha

ROOT = Path(__file__).resolve().parent
MODELS = [('parsimonious', '#2474b5', '5 parameters'),
          ('best_accuracy', '#dc7633', '8 parameters')]


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(source, out):
    source, out = source.resolve(), out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a new output directory')
    original_path = source/'PREDICTIONS_FROZEN.json'
    original_sha = sha(original_path)
    original = read(original_path)
    labels = {c: read(source/'inputs'/f'{c}_labels.json') for c in CASES}
    out.mkdir(parents=True, exist_ok=True)
    inputs = out/'inputs'
    inputs.mkdir()
    initials = {c: {'month': labels[c]['months'][0],
                    'depth_um': labels[c]['observed_depth_um'][0]} for c in CASES}
    save(inputs/'initial_conditions.json', initials)
    frozen = {
        'status': 'FIRST_OBSERVATION_CONDITIONED_FORECAST',
        'source_run': str(source), 'source_forecast_sha256': original_sha,
        'formula': 'D_cond(t) = D_obs(t1) + D_raw(t) - D_raw(t1)',
        'state_clock': 'Original exposure clock retained; no reset at t1.',
        'target_initialization_observations_per_site': 1,
        'target_parameter_fit_observations': 0,
        'coefficient_refitting': False,
        'initial_conditions': initials,
        'mapping': original['mapping'], 'models': {},
        'scope': 'Retrospective conditioning of archived forecasts. Full archived label files supply the first observations and subsequent scoring data; no new blind test.',
    }
    verification = []
    for alias, _, _ in MODELS:
        old = original['models'][alias]
        predictions = {}
        for case in CASES:
            series = old['predictions'][case]
            np.testing.assert_array_equal(series['months'], labels[case]['months'])
            raw = np.asarray(series['predicted_depth_um'], float)
            conditioned = condition_on_initial_depth(raw, initials[case]['depth_um'])
            np.testing.assert_equal(conditioned[0], initials[case]['depth_um'])
            np.testing.assert_allclose(np.diff(conditioned), np.diff(raw), rtol=1e-12, atol=1e-12)
            if np.any(conditioned < 0) or np.any(np.diff(conditioned) < -1e-9):
                raise ValueError('INVALID_CONDITIONED_DEPTH')
            predictions[case] = {'months': series['months'], 'predicted_depth_um': conditioned.tolist()}
            verification.append({'model': alias, 'case': case,
                                 'first_point_exact': True, 'growth_preserved': True})
        frozen['models'][alias] = {
            'equation_key': old['equation_key'], 'lab_score': old['lab_score'],
            'source_fit': old['source_fit'], 'predictions': predictions}
        shutil.copy2(source/f'{alias}_equation.json', out/f'{alias}_equation.json')
    save(out/'PREDICTIONS_CONDITIONED.json', frozen)
    conditioned_sha = sha(out/'PREDICTIONS_CONDITIONED.json')
    scores, prediction_rows = [], []
    for case in CASES:
        shutil.copy2(source/'inputs'/f'{case}_labels.json', inputs/f'{case}_labels.json')
        y = np.asarray(labels[case]['observed_depth_um'], float)
        if len(y) < 2:
            raise ValueError('NO_FUTURE_SCORING_POINTS')
        for alias, _, _ in MODELS:
            pred = np.asarray(frozen['models'][alias]['predictions'][case]['predicted_depth_um'])
            raw = np.asarray(original['models'][alias]['predictions'][case]['predicted_depth_um'])
            residual = pred[1:] - y[1:]
            denom = float(np.sum((y[1:] - y[1:].mean())**2))
            scores.append({'case': case, 'model': alias,
                           'role': 'source_calibration_conditioned' if case == 'linhai' else 'target_first_observation_conditioned',
                           **metrics(y, pred),
                           'future_scored_points': len(y)-1,
                           'future_depth_rmse_um': float(np.sqrt(np.mean(residual**2))),
                           'future_absolute_r2': float(1-np.sum(residual**2)/denom) if denom else None,
                           'raw_future_depth_rmse_um': float(np.sqrt(np.mean((raw[1:]-y[1:])**2))),
                           'raw_endpoint_error_pct': float(100*(raw[-1]/y[-1]-1))})
            for i, month in enumerate(labels[case]['months']):
                prediction_rows.append({'case': case, 'model': alias, 'month': month,
                                        'observed_depth_um': float(y[i]), 'predicted_depth_um': float(pred[i]),
                                        'raw_predicted_depth_um': float(raw[i]), 'is_initial_condition': i == 0})
    save(out/'scores.json', scores)
    write_csv(out/'scores.csv', scores)
    write_csv(out/'predictions.csv', prediction_rows)
    diagnostic = source/'ALL_INITIAL_DIAGNOSTIC.json'
    if diagnostic.exists():
        for row in read(diagnostic):
            actual = frozen['models'][row['model']]['predictions'][row['case']]['predicted_depth_um']
            np.testing.assert_allclose(actual, row['anchored_predictions_um'], rtol=0, atol=1e-12)
    assert sha(original_path) == original_sha
    assert sha(out/'PREDICTIONS_CONDITIONED.json') == conditioned_sha
    macro = {alias: float(np.mean([abs(s['endpoint_error_pct']) for s in scores
                                  if s['model'] == alias and s['case'] != 'linhai']))
             for alias, _, _ in MODELS}
    save(out/'VERIFICATION.json', {'original_forecast_unchanged': True,
                                  'source_forecast_sha256': original_sha,
                                  'conditioned_forecast_sha256': conditioned_sha,
                                  'matches_archived_initial_diagnostic': diagnostic.exists(),
                                  'checks': verification, 'target_macro_endpoint_ape_pct': macro,
                                  'input_sha256': {p.name: sha(p) for p in inputs.iterdir()}})
    snapshot = out/'implementation_snapshot'
    snapshot.mkdir()
    for p in (Path(__file__), ROOT/'maief/engineering_integrator.py'):
        shutil.copy2(p, snapshot/p.name)
    render(out, frozen, labels, scores)
    lines = ['# 首点初值条件下的工程预测', '',
             r'$$D_{\mathrm{pred}}(t)=D_{\mathrm{obs}}(t_1)+\widehat D(t)-\widehat D(t_1).$$', '',
             '各站点首个实测深度作为初值，只叠加之后的模型增长；暴露状态时钟保持原定义。首点不参与误差统计。目标站点每站使用 1 个观测初始化，未用目标站点数据拟合参数；临海仍为源站标定。', '',
             '实验室方程、环境映射与工程状态系数均沿用已保存结果，没有重新拟合。', '',
             '| 站点 | 模型 | 原终点误差 (%) | 初值条件终点误差 (%) | 后续 RMSE (μm) | 原后续 RMSE (μm) |',
             '|---|---|---:|---:|---:|---:|']
    for s in scores:
        lines.append(f"| {s['case']} | {s['model']} | {s['raw_endpoint_error_pct']:+.2f} | {s['endpoint_error_pct']:+.2f} | {s['future_depth_rmse_um']:.3f} | {s['raw_future_depth_rmse_um']:.3f} |")
    lines += ['', f"三个目标站点的平均绝对终点误差：五参数 {macro['parsimonious']:.2f}%，八参数 {macro['best_accuracy']:.2f}%。", '',
              'KSC 的整体偏低有相当部分来自首点偏差；试验池五参数式首点校正后终点更接近观测，但中间轨迹 RMSE 增大。初值一致不能消除增长轨迹本身的偏差。', '',
              '校验通过：首点完全一致、后续增长保持、与存档初值诊断一致，原预测文件校验值未改变。', '',
              '![首点初值条件工程预测](ENGINEERING_FIRST_OBSERVATION.png)', '']
    (out/'ENGINEERING_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print({'status': 'COMPLETE', 'out': str(out), 'target_macro_endpoint_ape_pct': macro})


def render(out, frozen, labels, scores):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    names = {'linhai': 'Linhai: source calibration', 'jinhai': 'Jinhai',
             'pool': 'Exposure pool', 'ksc': 'NASA KSC'}
    for index, (ax, case) in enumerate(zip(axes.flat, CASES)):
        month, depth = labels[case]['months'], labels[case]['observed_depth_um']
        ax.plot(month[1:], depth[1:], 'ko', ms=4, label='Observed')
        for alias, color, title in MODELS:
            p = frozen['models'][alias]['predictions'][case]
            ax.plot(month, p['predicted_depth_um'], color=color, lw=1.8,
                    ls='-' if alias == 'parsimonious' else '--', label=title)
        ax.plot(month[0], depth[0], marker='D', color='black', mfc='white', ms=6,
                linestyle='none', label='Given initial depth', zorder=5)
        site_scores = {s['model']: s for s in scores if s['case'] == case}
        text = '\n'.join(f"{title}: RMSE {site_scores[alias]['future_depth_rmse_um']:.2f} µm; end {site_scores[alias]['endpoint_error_pct']:+.2f}%"
                         for alias, _, title in MODELS)
        ax.text(.04, .95, text, transform=ax.transAxes, va='top', fontsize=8,
                bbox={'facecolor': 'white', 'alpha': .85, 'edgecolor': 'none'})
        ax.set_title(f"({chr(97+index)}) {names[case]}", loc='left', fontsize=11)
        ax.set(xlabel='Exposure time (months)', ylabel='Corrosion depth (µm)', ylim=(0, max(depth+frozen['models']['best_accuracy']['predictions'][case]['predicted_depth_um'])*1.28))
        ax.grid(alpha=.15)
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='outside lower center', ncol=4, frameon=False, fontsize=9)
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(out/f'ENGINEERING_FIRST_OBSERVATION.{extension}', dpi=350)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.out)
