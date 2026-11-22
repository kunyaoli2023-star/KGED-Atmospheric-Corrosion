"""Grounded feedback and hypothesis -> mathematics -> evaluation provenance.

Facts are computed by local code. Interpretation is still a model judgment.
Numerical response probes are not formal/global proofs, and development-CV
comparisons do not establish a corrosion mechanism or an independent test.
"""
from __future__ import annotations
import copy
import itertools
import json
import re
from pathlib import Path
import numpy as np
import sympy as sp
from scipy.optimize import brentq
from . import construction as base
from .storage import fingerprint, read, save, sha

PROTOCOL = 'GROUNDED_SCIENTIFIC_FEEDBACK_2_TYPED_REFERENCES'
RULES = [
    'Separate supplied computed facts, scientific interpretation, and a proposed prediction. Never invent a measured value, formula, fit or independent test.',
    'Refer to fact_ids for numerical/formula facts. Do not type formulas or measured/score numbers into narrative; code will display exact values with scope. fact_checks uses a JSON Pointer field_path inside the fact value (e.g. /score/cv_log_rmse or /modules/0/expression) and copies ONLY that field as valid JSON. Never summarize or omit keys in a checked value.',
    'Typed references: fact_ids contains ONLY supplied F_ computed-fact IDs; literature_ids contains ONLY supplied extraction IDs from available_evidence. A literature motif is guidance, not a measured/formula fact. An equation/module ID belongs to hypothesis evidence_ids or must be cited via its enclosing F_ fact, never placed in fact_ids. fact_checks can check ONLY F_ facts.',
    'An H3 or H4 module slot is not a physical property. Inspect complete-equation response_properties: an H4 closure can express an H3 film maximum. A fitted maximum or mixed derivative is model behavior, not experimental proof.',
    'Training residual aggregates are not observed response slopes. Conditional observed-trend counts are descriptive, not statistical significance. Grouped OOF residuals reuse observations across axes; adaptive CV is not untouched testing.',
    'Use EVIDENCE_REFRESH with the previous claim VERBATIM and new_observable_prediction EXACTLY the empty string when updating current reason/facts; do not restate the existing prediction in that field. A new incumbent or unresolved physical truth alone does not revise the claim. Actual status changes still require review.',
    'Use CLAIM_REVISION only with an explicit changed observable prediction and conditions. Translation, adjectives, or a new test suggestion alone are not a claim revision.',
    'Executable challenges must name the supplied question/contrast and cite supplied facts when available; put new measurement collection in future_experiments, not in executable_actions. Do not declare those experiments completed.',
    'Only registered canonical Q_ contrast IDs may be linked to mathematical proposals. A new contrast is a proposal, not confirmation; failed numerical quality gates are not physical falsification.',
    'Read all four PARENT_QUESTION_COVERAGE facts before calling any question untested. Their counts cover archived complete-equation evaluations, not a proof of the proposed mechanism. A pre-challenge interpretation does not describe a challenger fitted later in the same cycle.',
    'Distinguish the lowest-CV equation from the simplicity-tolerant default using SELECTION_STATE. Do not call an archived form newly discovered or a new incumbent without an explicit prior-archive comparison.',
]


def extend_schema(schema):
    schema = copy.deepcopy(schema)
    fact_refs = base.arr(dict(base.STR, description='ONLY computed-fact IDs from evidence_snapshot.facts.'))
    literature_refs = base.arr(dict(base.STR, description='ONLY extraction IDs from available_evidence; not numerical evidence.'))
    schema['properties'].update({
        'fact_checks': base.arr(base.obj({'fact_id': base.STR, 'field_path':base.STR,
            'expected_value_json': dict(base.STR,description='Valid JSON encoding of exactly the addressed field. A string expression needs JSON double quotes, e.g. "k*log(tau)"; not a bare expression.')})),
        'analyses': base.arr(base.obj({'analysis_id': base.STR,
            'question_ids': base.arr(base.enum(['H1','H2','H3','H4'])),
            'fact_ids': fact_refs, 'literature_ids': literature_refs,
            'interpretation': base.STR, 'limits': base.STR})),
        'parent_update_contracts': base.arr({'anyOf':[
            base.obj({'hypothesis_id':base.enum(['H1','H2','H3','H4']),
                'update_kind':base.enum(['EVIDENCE_REFRESH']),
                'new_observable_prediction':base.enum(['']), 'fact_ids':fact_refs,
                'literature_ids':literature_refs}),
            base.obj({'hypothesis_id':base.enum(['H1','H2','H3','H4']),
                'update_kind':base.enum(['CLAIM_REVISION']),
                'new_observable_prediction':dict(type='string',minLength=1), 'fact_ids':fact_refs,
                'literature_ids':literature_refs})]}),
        'executable_actions': base.arr(base.obj({'question_id': base.enum(['H1','H2','H3','H4']),
            'contrast_ids': base.arr(base.STR), 'fact_ids': fact_refs, 'literature_ids': literature_refs,
            'mathematical_change_to_test': base.STR, 'matched_control': base.STR,
            'discriminating_prediction': base.STR})),
        'future_experiments': base.arr(base.STR),
    })
    schema['required'] += ['fact_checks','analyses','parent_update_contracts',
                           'executable_actions','future_experiments']
    return schema


class FeedbackValidationError(ValueError):
    """Actionable reference error, without exposing unsupplied/private data."""
    def __init__(self, code, path, invalid_ids, namespace):
        self.details = dict(code=code, field_path=path, invalid_ids=sorted(invalid_ids),
                            expected_namespace=namespace)
        super().__init__(code+' '+json.dumps(self.details,ensure_ascii=False))


def reference_catalogue(payload):
    return dict(computed_fact_ids=sorted(f['fact_id'] for f in payload['evidence_snapshot']['facts']),
        literature_extraction_ids=sorted(e['evidence_id'] for e in payload.get('available_evidence',[])),
        rule='Exact supplied IDs only; these namespaces are not interchangeable scientific evidence.')


def bind_fact_check_schema(schema,payload):
    """Constrain exact checks to actual fact IDs, not invented check labels."""
    bound=copy.deepcopy(schema)
    ids=reference_catalogue(payload)['computed_fact_ids']
    if ids:
        bound['properties']['fact_checks']['items']['properties']['fact_id']=dict(
            base.enum(ids),description='The fact_id of the supplied computed fact; NEVER a new check identifier such as FC1.')
    return bound


def normalize_references(answer, payload):
    """Lossless representation fixes only; never repair a value or scientific claim.

    The raw response stays immutable. Missing legacy literature arrays mean no
    such citations, not additional evidence. fact_checks is deliberately untouched.
    """
    value=copy.deepcopy(answer);changes=[]
    known=set(reference_catalogue(payload)['literature_extraction_ids'])
    for group in ('analyses','parent_update_contracts','executable_actions'):
        for index, row in enumerate(value[group]):
            path=f'/{group}/{index}'
            if 'literature_ids' not in row:
                row['literature_ids']=[]
                changes.append(dict(field_path=path+'/literature_ids',rule='LEGACY_EMPTY_NAMESPACE',ids=[]))
            routed=[r for r in row['fact_ids'] if r in known]
            if routed:
                row['fact_ids']=[r for r in row['fact_ids'] if r not in known]
                row['literature_ids']=list(dict.fromkeys(row['literature_ids']+routed))
                changes.append(dict(field_path=path+'/fact_ids',to_field_path=path+'/literature_ids',
                    rule='SUPPLIED_EXTRACTION_ID_NAMESPACE_ROUTING_ONLY',ids=routed))
    facts={f['fact_id']:f for f in payload['evidence_snapshot']['facts']}
    for index,check in enumerate(value['fact_checks']):
        if check['fact_id'] not in facts:
            continue
        try:
            json.loads(check['expected_value_json'])
        except (ValueError,TypeError):
            try:actual=fact_field(facts[check['fact_id']]['value'],check['field_path'])
            except ValueError:continue
            # Only an exactly matching supplied STRING with missing JSON quotes
            # is serialized. Unknown/wrong expressions, numbers or partial
            # objects are never repaired to match the recorded value.
            if isinstance(actual,str) and check['expected_value_json']==actual:
                check['expected_value_json']=json.dumps(actual,ensure_ascii=False)
                changes.append(dict(field_path=f'/fact_checks/{index}/expected_value_json',
                    rule='EXACT_SUPPLIED_STRING_JSON_ENCODING_ONLY',fact_id=check['fact_id'],
                    checked_field_path=check['field_path']))
    return value,changes


def complete_unchanged_claim_contracts(answer, payload):
    """Fill missing bookkeeping only from the model's own typed evidence.

    This cannot revise a claim or supply an observable prediction. If an
    evidentially changed parent has no typed fact or extraction citation, the
    normal scientific validator must still reject the answer.
    """
    value=copy.deepcopy(answer)
    before={h['hypothesis_id']:h for h in payload['hypotheses_before']}
    states={h['hypothesis_id']:h for h in value['hypotheses']}
    if set(states)!=set(before) or len(value['hypotheses'])!=len(before):
        return value,[]
    existing=value['parent_update_contracts']
    supplied=[c['hypothesis_id'] for c in existing]
    if len(set(supplied))!=len(supplied) or not set(supplied)<=set(before):
        return value,[]
    missing=sorted(set(before)-set(supplied))
    if not missing:
        return value,[]
    facts={f['fact_id'] for f in payload['evidence_snapshot']['facts']}
    literature={e['evidence_id'] for e in payload.get('available_evidence',[])}
    contracts=[]
    for hid in missing:
        old,new=before[hid],states[hid]
        if new['claim']!=old['claim']:
            return value,[]
        cited=list(dict.fromkeys(new['evidence_ids']))
        fact_ids=[ref for ref in cited if ref in facts]
        literature_ids=[ref for ref in cited if ref in literature]
        if (new['reason']!=old.get('reason') or new['status']!=old['status']) and not (
                fact_ids or literature_ids):
            return value,[]
        contracts.append(dict(hypothesis_id=hid,update_kind='EVIDENCE_REFRESH',
            new_observable_prediction='',fact_ids=fact_ids,
            literature_ids=literature_ids))
    value['parent_update_contracts']=existing+contracts
    return value,[dict(field_path='/parent_update_contracts',
        rule='DERIVED_UNCHANGED_CLAIM_CONTRACTS_FROM_MODEL_TYPED_EVIDENCE_ONLY',
        hypothesis_ids=missing,scientific_claim_or_prediction_added=False)]


def remove_retyped_narrative_literals(answer, payload):
    """Drop unverified literal restatements, never their cited source facts.

    A narrative without an explicit supplied F_ citation is left untouched for
    the ordinary validator to reject. Exact expressions and scores remain in
    the immutable raw response and the structured computed-fact ledger.
    """
    value=copy.deepcopy(answer)
    known={f['fact_id'] for f in payload['evidence_snapshot']['facts']}
    changes=[]
    factual=re.compile(r'\b(?:log1p|log|exp|sqrt|tanh)\s*\(|\b(?:tau|c|h)\s*\*\*|\b\d+\.\d{3,}\b')
    literal_number=re.compile(r'\b\d+\.\d{3,}\b')
    backticked=re.compile(r'`([^`]+)`')

    def clean(text,path):
        cited=sorted(set(re.findall(r'F_[0-9a-f]+',text)) & known)
        if not cited or not factual.search(text):
            return text
        result=backticked.sub(lambda match: '`[see cited structured expression]`'
            if factual.search(match.group(1)) else match.group(0),text)
        result=literal_number.sub('[see cited structured value]',result)
        if result!=text:
            changes.append(dict(field_path=path,
                rule='REMOVE_RETYPE_KEEP_EXPLICIT_SUPPLIED_FACT_CITATION',
                cited_fact_ids=cited,original_text_sha256=fingerprint(text),
                scientific_claim_or_prediction_added=False))
        return result

    value['round_summary']=clean(value['round_summary'],'/round_summary')
    for group in ('diagnosis','optimization_plan'):
        value[group]=[clean(text,f'/{group}/{i}') for i,text in enumerate(value[group])]
    for group,field in (('hypotheses','reason'),('analyses','interpretation')):
        for i,row in enumerate(value[group]):
            row[field]=clean(row[field],f'/{group}/{i}/{field}')
    return value,changes


def require_references(ids, allowed, path, namespace, code):
    invalid=set(ids)-set(allowed)
    if invalid:
        raise FeedbackValidationError(code,path,invalid,namespace)


def fact_field(value, pointer):
    """Exact JSON Pointer traversal; no expression evaluation or fuzzy subset."""
    if pointer=='': return value
    if not pointer.startswith('/'): raise ValueError('FACT_CHECK_REQUIRES_JSON_POINTER')
    for token in pointer[1:].split('/'):
        token=token.replace('~1','/').replace('~0','~')
        if isinstance(value,list):
            if not token.isdigit() or int(token)>=len(value): raise ValueError('FACT_CHECK_UNKNOWN_FIELD')
            value=value[int(token)]
        elif isinstance(value,dict) and token in value: value=value[token]
        else: raise ValueError('FACT_CHECK_UNKNOWN_FIELD')
    return value


def fact(kind, value, phase, scope):
    body = dict(kind=kind, value=value, phase=phase, scope=scope)
    return dict(fact_id='F_'+fingerprint(body)[:20], **body)


def data_summary(inputs, default):
    """No individual response labels are placed in the inference payload."""
    table = read(inputs/'data.json'); folds = read(inputs/'folds.json')
    y = np.asarray(table['log_i'], float)
    axes = ('T_C','C_wt','h_um')
    descriptions = {}
    for axis in axes:
        other = [k for k in axes if k != axis]
        curves = {}
        for i in range(len(y)):
            curves.setdefault(tuple(table[k][i] for k in other), {}).setdefault(table[axis][i], []).append(y[i])
        counts = dict(increase=0, decrease=0, numerically_equal=0, conditioning_slices=0,
                      slices_with_both_signs=0)
        for levels in curves.values():
            if len(levels)<2: continue
            counts['conditioning_slices'] += 1
            means = [float(np.mean(levels[k])) for k in sorted(levels)]
            delta = np.diff(means)
            counts['increase'] += int(np.sum(delta>1e-10))
            counts['decrease'] += int(np.sum(delta<-1e-10))
            counts['numerically_equal'] += int(np.sum(np.abs(delta)<=1e-10))
            counts['slices_with_both_signs'] += int(np.any(delta>1e-10) and np.any(delta<-1e-10))
        descriptions[axis] = dict(levels=sorted(set(table[axis])), adjacent_observed_log_response=counts)
    oof = {}
    for row in default.get('fold_rows', []):
        residual = np.asarray(row['predicted'])-np.asarray(row['observed'])
        oof.setdefault(row['axis'], []).extend(residual.tolist())
    return dict(observations=len(y), folds=len(folds),
        data_sha256=sha(inputs/'data.json'), folds_sha256=sha(inputs/'folds.json'),
        units={'T_C':'degC','C_wt':'NaCl wt%','h_um':'liquid electrolyte um',
               'response':'natural-log corrosion current in the input current unit'},
        coordinates={'tau':'(T_C+273.15)/298.15','c':'C_wt/(1 wt%)','h':'h_um/(200 um)'},
        observed_conditional_trends=descriptions,
        oof_residuals_by_grouping_axis={axis:dict(validation_appearances=len(values),
            mean_log_residual=float(np.mean(values)), rmse=float(np.sqrt(np.mean(np.square(values)))))
            for axis,values in oof.items()},
        scope='Fixed data/folds. Adjacent trends average replicates at identical other coordinates; the tolerance is numerical resolution, NOT significance. OOF appearances across grouping axes are not independent observations. No raw response table or field label is sent.')


def response_properties(equation, protocol, points=129):
    """Differentiate the complete equation; probe declared-domain slices only.

    Root brackets can miss tangent roots or closely spaced extrema. Unresolved
    boundaries/conditions remain explicit. No trend is forced by an H label.
    """
    symbols = {v:sp.Symbol(v, real=True) for v in ('tau','c','h')}
    expression = sp.Integer(0)
    for i,m in enumerate(equation.get('physical_modules', [])):
        params = {name:sp.Symbol('m%d_%s'%(i,name), real=True) for name in m['parameters']}
        mapping = {params[k]:float(v) for k,v in m['parameters'].items()}
        expression += base.symbolic(m['expression'], {**symbols,**params}).subs(mapping)
    slices = []; mixed = {}
    for axis in ('tau','c','h'):
        derivative = sp.diff(expression, symbols[axis])
        fn = sp.lambdify([symbols[v] for v in ('tau','c','h')], derivative, 'numpy', cse=True)
        others = [v for v in ('tau','c','h') if v != axis]
        contexts = [dict(protocol['reference'])]
        for values in itertools.product(*(protocol['domains'][v] for v in others)):
            contexts.append(dict(protocol['reference'], **dict(zip(others,values))))
        for context in contexts:
            grid = np.linspace(*protocol['domains'][axis], points)
            coordinates = {v:np.full(points,context[v]) for v in symbols}; coordinates[axis]=grid
            try:
                with np.errstate(all='ignore'):
                    slope = np.broadcast_to(np.asarray(fn(*(coordinates[v] for v in symbols)),float), grid.shape)
                if not np.isfinite(slope).all(): raise ValueError('NONFINITE_DERIVATIVE_OR_BOUNDARY')
                tolerance = 1e-9*max(1.,float(np.max(np.abs(slope))))
                positive = bool(np.any(slope>tolerance)); negative = bool(np.any(slope<-tolerance))
                trend = 'NONMONOTONE' if positive and negative else 'INCREASING' if positive else 'DECREASING' if negative else 'CONSTANT_AT_PROBE_RESOLUTION'
                roots=[]
                def scalar(x):
                    values={**context,axis:float(x)}
                    return float(fn(*(values[v] for v in symbols)))
                # Include a sampled zero only when signs on its two sides differ.
                for j in range(points-1):
                    if slope[j]*slope[j+1]<0:
                        root=float(brentq(scalar,float(grid[j]),float(grid[j+1])))
                        roots.append(dict(coordinate=root, kind='MAXIMUM' if slope[j]>0 else 'MINIMUM'))
                    elif j and abs(slope[j])<=tolerance and slope[j-1]*slope[j+1]<0:
                        roots.append(dict(coordinate=float(grid[j]),kind='MAXIMUM' if slope[j-1]>0 else 'MINIMUM'))
                slices.append(dict(axis=axis, fixed={v:context[v] for v in others},
                    probed_domain=protocol['domains'][axis], trend=trend, stationary_crossings=roots))
            except (ValueError,TypeError,OverflowError,ZeroDivisionError) as exc:
                slices.append(dict(axis=axis,fixed={v:context[v] for v in others},
                    trend='UNRESOLVED_NUMERICAL_PROBE',error=str(exc)[:120]))
        if axis=='h':
            for other in others:
                cross=sp.diff(derivative,symbols[other])
                mixed[other]=dict(symbolically_zero=bool(cross==0),
                    scope='Derivative of the fitted complete expression; nonzero does not prove measured or causal coupling.')
    return dict(equation_key=equation['equation_key'], conditional_slices=slices,
        mixed_film_environment_derivatives=mixed, grid_points_per_slice=points,
        parameter_and_response_scope='Full-data fitted coefficients; centred offsets and intercept do not change these derivatives. Formal construction conditions were checked separately at admission/fitting. These slices do NOT prove conditions or trends everywhere.',
        global_or_physical_proof=False)


def module_links(row, by_id):
    """Exact fusions inherit edges, not a new physical hypothesis."""
    questions=list(row['proposal'].get('question_links', []))
    contrasts=list(row['proposal'].get('contrast_links', []))
    for annotation in row.get('lineage_annotations', []):
        questions += annotation.get('question_links', [])
        contrasts += annotation.get('contrast_links', [])
    for mid in row.get('parent_modules', []):
        if mid in by_id:
            q,c=module_links(by_id[mid],by_id); questions+=q; contrasts+=c
    return questions,contrasts


def test_graph(pool, hypotheses, records):
    """A test ledger, never an automatic confirmation/falsification verdict."""
    by_id={r['module_id']:r for r in pool}; edges=[]
    for row in pool:
        questions,links=module_links(row,by_id)
        for link in links:
            edges.append(dict(contrast_id=link['contrast_id'],module_id=row['module_id'],
                parent_questions=sorted({q['hypothesis_id'] for q in questions}),
                relation=link['relation'],observable_prediction=link['observable_prediction']))
    children=[]
    for h in hypotheses:
        mids=sorted({e['module_id'] for e in edges if e['contrast_id']==h['new_hypothesis_id']})
        evaluations=[r for r in records if set(r['modules'])&set(mids)]
        scored=[r for r in evaluations if r.get('score')]
        eligible=[r for r in scored if r.get('admissible')]
        if not mids: status='REGISTERED_NO_LINKED_IMPLEMENTATION'
        elif not evaluations: status='IMPLEMENTED_NOT_FULLY_EVALUATED'
        elif not scored: status='EVALUATED_NUMERICALLY_UNRESOLVED'
        elif not eligible: status='TESTED_NO_ELIGIBLE_IMPLEMENTATION'
        else: status='ELIGIBLE_IMPLEMENTATIONS_TESTED_NOT_CONFIRMATION'
        comparison_focus=sorted(scored,key=lambda r:(not r.get('admissible'),r['score']['cv_log_rmse'],r['equation_key']))[:8]
        comparisons=[c for r in comparison_focus for c in matched_comparisons(r,records,pool,limit=4)
            if not (set(c['control_only_modules'])&set(mids))]
        if eligible and not any(c['focus_admissible'] and c['control_admissible'] for c in comparisons):
            status='ELIGIBLE_TESTED_NO_ELIGIBLE_MATCHED_CONTROL'
        children.append(dict(contrast_id=h['new_hypothesis_id'],parent=h['parent_hypothesis_id'],
            claim=h['claim'],prediction=h['testable_prediction'],falsification=h['falsification'],
            module_ids=mids,test_status=status,
            matched_development_comparisons=comparisons,
            comparison_focus_equations=[r['equation_key'] for r in comparison_focus],
            comparison_neighborhood_exhausted=False,
            evaluations=[dict(equation_key=r['equation_key'],modules=r['modules'],
                score=r.get('score'),admissible=bool(r.get('admissible')),
                numerical_error=r.get('error',r.get('error_class'))) for r in evaluations],
            scope='These registered links are proposed correspondences, reviewed with the construction. Numerical failure or CV loss does not physically falsify the contrast; no new experiment is claimed.'))
    return dict(protocol=PROTOCOL,edges=edges,contrasts=children,
        module_questions=[dict(module_id=r['module_id'], question_links=module_links(r,by_id)[0],
            explicit_mapping=bool(module_links(r,by_id)[0])) for r in pool])


def parent_question_coverage(pool, records, default_key):
    """Count evaluated complete equations linked to each declared question.

    These are search-coverage counts, not support, significance or physical
    validation. Rival controls are reported separately from implementations.
    """
    by_id={row['module_id']:row for row in pool}
    records=[row for row in records if set(row['modules'])<=set(by_id)]
    links={mid:module_links(row,by_id)[0] for mid,row in by_id.items()}
    symbols={name:sp.Symbol(name,real=True) for name in ('tau','c','h')}
    coupled=set()
    for mid,row in by_id.items():
        proposal=row['proposal']
        if not any(e.get('hypothesis_id')=='H4' for e in links[mid]):
            continue
        parameters={item['name']:sp.Symbol(item['name'],real=True)
            for item in proposal['parameters']}
        expression=base.symbolic(proposal['expression'],{**symbols,**parameters})
        if any(sp.diff(expression,symbols['h'],symbols[other])!=0 for other in ('tau','c')):
            coupled.add(mid)
    result=[]
    for hid in ('H1','H2','H3','H4'):
        kinds=('IMPLEMENTATION','RIVAL_CONTROL')
        modules={kind:sorted(mid for mid,edges in links.items()
            if any(e.get('hypothesis_id')==hid and e.get('relation')==kind for e in edges))
            for kind in kinds}
        linked=set(modules['IMPLEMENTATION'])|set(modules['RIVAL_CONTROL'])
        evaluated=[row for row in records if set(row['modules'])&linked]
        implementation=[row for row in records if set(row['modules'])&set(modules['IMPLEMENTATION'])]
        rival=[row for row in records if set(row['modules'])&set(modules['RIVAL_CONTROL'])]
        result.append(dict(hypothesis_id=hid,linked_module_count=len(linked),
            implementation_module_count=len(modules['IMPLEMENTATION']),
            rival_control_module_count=len(modules['RIVAL_CONTROL']),
            evaluated_complete_equation_count=len(evaluated),
            scored_complete_equation_count=sum(bool(row.get('score')) for row in evaluated),
            admissible_complete_equation_count=sum(bool(row.get('admissible')) for row in evaluated),
            implementation_scored_count=sum(bool(row.get('score')) for row in implementation),
            rival_control_scored_count=sum(bool(row.get('score')) for row in rival),
            structurally_coupled_module_count=len(linked&coupled),
            structurally_coupled_scored_equation_count=sum(bool(row.get('score')) and
                bool(set(row['modules'])&coupled) for row in evaluated),
            default_contains_linked_module=any(row['equation_key']==default_key for row in evaluated),
            scope='Archived full-validation complete equations linked by declared module-question edges. Structural coupling means a nonzero symbolic h-T or h-c mixed derivative before fitting; a zero fitted coefficient can still remove its numerical effect. A linked module or score does not verify the hypothesis; absence of a link does not prove absence of the mathematical response.'))
    return result


def matched_comparisons(focus, records, pool, limit=8):
    """One-slot/ablation controls, not unmatched Pareto rankings or a p-value."""
    by_id={p['module_id']:p for p in pool}
    def slot(mid):
        p=by_id.get(mid,{})
        if p.get('assembly_scope')=='COMPLETE_EQUATION': return 'COMPLETE'
        hid=p.get('proposal',{}).get('hypothesis_id')
        return 'FILM' if hid in ('H3','H4') else hid
    result=[]
    for row in records:
        if row['equation_key']==focus['equation_key'] or not row.get('score'): continue
        a=set(focus['modules'])-set(row['modules']); b=set(row['modules'])-set(focus['modules'])
        if len(a)>1 or len(b)>1 or (not a and not b): continue
        if any(slot(m) in (None,'COMPLETE') for m in a|b): continue
        if a and b and slot(next(iter(a)))!=slot(next(iter(b))): continue
        if row['score'].get('validation_folds')!=focus['score'].get('validation_folds'): continue
        result.append(dict(focus_equation_key=focus['equation_key'],control_equation_key=row['equation_key'],
            shared_module_ids=sorted(set(row['modules'])&set(focus['modules'])),
            focus_only_modules=sorted(a),control_only_modules=sorted(b),
            focus_cv=float(focus['score']['cv_log_rmse']),control_cv=float(row['score']['cv_log_rmse']),
            delta_focus_minus_control=float(focus['score']['cv_log_rmse']-row['score']['cv_log_rmse']),
            focus_admissible=bool(focus.get('admissible')),control_admissible=bool(row.get('admissible')),
            scope='Same companion module identities and grouped folds; all coefficients refitted, not held equal. Ineligible fits are descriptive only. Adaptive development comparison, not significance or a physical verdict.'))
    return sorted(result,key=lambda r:(not r['control_admissible'],abs(r['delta_focus_minus_control']),r['control_equation_key']))[:limit]


def evidence_snapshot(inputs, protocol, pool, hypotheses, records, archive, default,
                      phase='PRE_CHALLENGE_FEEDBACK'):
    representatives={r['equation_key']:r for r in sorted(archive,key=lambda r:r['score']['cv_log_rmse'])[:6]}
    representatives[default['equation_key']]=default
    facts=[fact('DATA_CONTRACT',data_summary(inputs,default),phase,'Descriptive fixed laboratory data, not target observations.')]
    best=min(archive,key=lambda row:(row['score']['cv_log_rmse'],row['equation_key']))
    facts.append(fact('SELECTION_STATE',dict(best_cv_equation_key=best['equation_key'],
        best_cv_log_rmse=best['score']['cv_log_rmse'],default_equation_key=default['equation_key'],
        default_cv_log_rmse=default['score']['cv_log_rmse'],
        default_is_best_cv=best['equation_key']==default['equation_key'],
        scope='Best-CV and simplicity-tolerant default are different selection roles; neither is an independent-test winner.'),
        phase,'Code-computed selection identities from the current full-CV archive.'))
    for coverage in parent_question_coverage(pool,records,default['equation_key']):
        facts.append(fact('PARENT_QUESTION_COVERAGE',coverage,phase,coverage['scope']))
    for r in representatives.values():
        value=dict(equation_key=r['equation_key'],selection_role=(
            'DEFAULT_AND_BEST_CV' if r['equation_key']==default['equation_key']==best['equation_key'] else
            'DEFAULT' if r['equation_key']==default['equation_key'] else
            'BEST_CV' if r['equation_key']==best['equation_key'] else 'PARETO_REPRESENTATIVE'),
            score=r['score'],modules=r['physical_modules'],intercept=r['intercept'],admissible=bool(r.get('admissible')))
        facts.append(fact('FITTED_EQUATION',value,phase,'Full grouped CV plus training fit; not mechanism proof or independent validation.'))
        facts.append(fact('RESPONSE_PROPERTIES',response_properties(r,protocol),phase,'Complete-equation fitted conditional derivative probes within the stated predictor domain.'))
    for row in matched_comparisons(default,records,pool):
        facts.append(fact('MATCHED_DEVELOPMENT_COMPARISON',row,phase,row['scope']))
    graph=test_graph(pool,hypotheses,records)
    for row in graph['contrasts']:
        # The complete graph retains every row locally. Do not dump all fold labels into inference.
        facts.append(fact('CONTRAST_TEST_STATUS',row,phase,row['scope']))
    return dict(protocol=PROTOCOL,phase=phase,default_equation_key=default['equation_key'],
        facts=facts,hypothesis_math_test_graph=graph,
        scope='Computed facts and declared/reviewed links are distinct from model interpretation; no experiment/physical proof is fabricated.')


def model_snapshot(snapshot, per_contrast=8):
    """Bound inference context; retain the complete provenance locally."""
    result=copy.deepcopy(snapshot)
    # The mathematical proposer separately receives the current module list.
    # Do not send a duplicate, ever-growing all-module graph to the analyst.
    result.pop('hypothesis_math_test_graph',None)
    for i,f in enumerate(result['facts']):
        if f['kind']!='CONTRAST_TEST_STATUS': continue
        value=f['value']; rows=value['evaluations']
        selected=sorted(rows,key=lambda r:(not r['admissible'],r.get('score') is None,
            r.get('score',{}).get('cv_log_rmse',float('inf')) if r.get('score') else float('inf'),r['equation_key']))[:per_contrast]
        value.update(evaluation_count=len(rows),eligible_count=sum(r['admissible'] for r in rows),
            evaluations=selected,provided_evaluation_count=len(selected),
            full_ledger_saved_locally=True)
        result['facts'][i]=fact(f['kind'],value,f['phase'],f['scope']+' Bounded row excerpt with total counts; not every equation is supplied to inference.')
    return result


def post_challenge_reconciliation(before, after):
    """Expose a changed selected equation without reusing pre-fit prose for it.

    The next scientific cycle must review the new fit. This comparison checks
    identities and computed response probes; it does not judge mechanisms.
    """
    def selected(snapshot):
        key=snapshot['default_equation_key']
        rows={f['value']['equation_key']:f['value'] for f in snapshot['facts']
            if f['kind']=='FITTED_EQUATION'}
        probes={f['value']['equation_key']:f['value'] for f in snapshot['facts']
            if f['kind']=='RESPONSE_PROPERTIES'}
        row=rows[key]; properties=probes[key]
        reference={}
        for axis in ('tau','c','h'):
            slices=[item for item in properties['conditional_slices'] if item['axis']==axis]
            reference[axis]=dict(trend=slices[0]['trend'],
                stationary_crossings=slices[0].get('stationary_crossings',[]))
        return dict(equation_key=key,score=row['score'],module_ids=[m['module_id'] for m in row['modules']],
            reference_response_probes=reference,
            mixed_film_environment_derivatives=properties['mixed_film_environment_derivatives'])
    first,last=selected(before),selected(after)
    changed=first['equation_key']!=last['equation_key']
    return dict(pre_challenge_default=first,post_challenge_default=last,
        selected_equation_changed=changed,
        reference_response_probe_changed=first['reference_response_probes']!=last['reference_response_probes'],
        next_cycle_scientific_review_required=changed,
        prior_feedback_scope='The model interpretation and parent-state audit preceded challenger fitting. They do not adjudicate a newly selected post-challenge equation.',
        scope='Deterministic fitted-expression comparison on declared-domain reference probes; no mechanism proof, target-label use, or unrecorded model call.')


def validate_feedback(value, payload):
    facts={f['fact_id']:f for f in payload['evidence_snapshot']['facts']}
    known_literature={e['evidence_id'] for e in payload.get('available_evidence',[])}
    old={h['hypothesis_id']:h for h in payload['hypotheses_before']}
    supplied={*facts,*known_literature}
    for f in facts.values():
        v=f['value']
        if isinstance(v,dict):
            for k in ('equation_key','focus_equation_key','control_equation_key'):
                if v.get(k): supplied.add(v[k])
            for m in v.get('modules',[]):
                if isinstance(m,dict): supplied.add(m['module_id'])
    supplied.update(m for r in payload.get('pareto_representatives',[]) for m in
        [r['equation_key']]+[p['module_id'] for p in r['physical_modules']])
    if sorted(h['hypothesis_id'] for h in value['hypotheses'])!=sorted(old):
        raise ValueError('FEEDBACK_EXACT_PARENT_COVERAGE_REQUIRED')
    contracts=value['parent_update_contracts']
    if sorted(c['hypothesis_id'] for c in contracts)!=sorted(old):
        raise ValueError('FEEDBACK_UPDATE_CONTRACT_COVERAGE_REQUIRED')
    states={h['hypothesis_id']:h for h in value['hypotheses']}
    for index,c in enumerate(contracts):
        hid=c['hypothesis_id']
        require_references(c['fact_ids'],facts,f'/parent_update_contracts/{index}/fact_ids',
            'COMPUTED_FACT', 'FEEDBACK_UNKNOWN_FACT_ID')
        require_references(c.get('literature_ids',[]),known_literature,
            f'/parent_update_contracts/{index}/literature_ids','LITERATURE_EXTRACTION','FEEDBACK_UNKNOWN_LITERATURE_ID')
        if (states[hid]['reason']!=old[hid].get('reason') or states[hid]['status']!=old[hid]['status']) and not (
                c['fact_ids'] or c.get('literature_ids') or set(states[hid]['evidence_ids'])&known_literature):
            raise ValueError('EVIDENCE_UPDATE_REQUIRES_CURRENT_FACTS_OR_SUPPLIED_EXTRACTIONS')
        if c['update_kind']=='EVIDENCE_REFRESH':
            if states[hid]['claim']!=old[hid]['claim'] or c['new_observable_prediction'].strip():
                raise ValueError('EVIDENCE_REFRESH_CANNOT_REWRITE_SCIENTIFIC_CLAIM_'+hid+'_use_verbatim_claim_and_empty_new_prediction')
        elif not c['new_observable_prediction'].strip() or states[hid]['claim']==old[hid]['claim']:
            raise ValueError('CLAIM_REVISION_REQUIRES_CHANGED_CLAIM_AND_OBSERVABLE_PREDICTION')
    for index,check in enumerate(value['fact_checks']):
        require_references([check['fact_id']],facts,f'/fact_checks/{index}/fact_id',
            'COMPUTED_FACT_EXACT_VALUE','FEEDBACK_UNKNOWN_FACT_ID')
        try: expected=json.loads(check['expected_value_json'])
        except (ValueError,TypeError):
            raise FeedbackValidationError('FACT_CHECK_VALUE_MUST_BE_JSON',
                f'/fact_checks/{index}/expected_value_json',[check['fact_id']],
                'EXACT_FIELD_AS_VALID_JSON') from None
        actual=fact_field(facts[check['fact_id']]['value'],check['field_path'])
        if fingerprint(expected)!=fingerprint(actual):
            raise ValueError('FACT_CHECK_CONTENT_MISMATCH_'+check['fact_id']+check['field_path'])
    for group in ('hypotheses','new_hypotheses'):
        for index,h in enumerate(value[group]):
            require_references(h['evidence_ids'],supplied,f'/{group}/{index}/evidence_ids',
                'SUPPLIED_EVIDENCE','FEEDBACK_UNKNOWN_EVIDENCE_ID')
    for group in ('analyses','executable_actions'):
        for index,a in enumerate(value[group]):
            require_references(a['fact_ids'],facts,f'/{group}/{index}/fact_ids',
                'COMPUTED_FACT','FEEDBACK_UNKNOWN_FACT_ID')
            require_references(a.get('literature_ids',[]),known_literature,f'/{group}/{index}/literature_ids',
                'LITERATURE_EXTRACTION','FEEDBACK_UNKNOWN_LITERATURE_ID')
    registered={h['new_hypothesis_id'] for h in payload.get('new_subhypotheses_before',[])}
    for a in value['executable_actions']:
        if not set(a['contrast_ids'])<=registered: raise ValueError('ACTION_REQUIRES_REGISTERED_CONTRAST_ID')
        if not all(a[k].strip() for k in ('mathematical_change_to_test','matched_control','discriminating_prediction')):
            raise ValueError('EXECUTABLE_ACTION_REQUIRES_CHANGE_CONTROL_AND_PREDICTION')
    # Structured references supply formulas/scores. Do not silently accept an
    # invented ASCII formula in factual narrative (the recorded salt error).
    narrative=[value['round_summary'],*value['diagnosis'],*value['optimization_plan'],
        *(h['reason'] for h in value['hypotheses']),
        *(a['interpretation'] for a in value['analyses'])]
    for text in narrative:
        if re.search(r'\b(?:log1p|log|exp|sqrt|tanh)\s*\(|\b(?:tau|c|h)\s*\*\*|\b\d+\.\d{3,}\b',text):
            raise ValueError('FACTUAL_NARRATIVE_MUST_REFERENCE_FACTS_NOT_RETYPE_FORMULAS_OR_SCORES')
    coverage={f['value']['hypothesis_id']:f['value'] for f in facts.values()
        if f['kind']=='PARENT_QUESTION_COVERAGE'}
    for text in narrative:
        for sentence in re.split(r'(?<=[.!?。！？])\s*',text):
            for hid,row in coverage.items():
                if not re.search(r'\b'+hid+r'\b',sentence,re.I) or re.search(
                        r'\b(experimentally|physical(?:ly)?|in new measurements)\b|实测|物理验证',sentence,re.I):
                    continue
                if row['scored_complete_equation_count'] and re.search(
                        r'\buntested\b|\bnot\s+(?:yet\s+)?(?:numerically\s+)?evaluated\b|\bno\s+(?:\w+\s+){0,10}(?:models?|equations?|candidates?)\s+(?:\w+\s+){0,10}evaluated\b|未(?:进行)?(?:数值)?(?:测试|评估)',
                        sentence,re.I):
                    raise ValueError('FEEDBACK_CONTRADICTS_EVALUATED_PARENT_COVERAGE_'+hid)
                if hid=='H4' and row['structurally_coupled_scored_equation_count'] and re.search(
                        r'\bno\s+(?:\w+\s+){0,8}interaction\s+(?:terms?|models?)\s+(?:\w+\s+){0,10}(?:evaluated|included)\b',
                        sentence,re.I):
                    raise ValueError('FEEDBACK_CONTRADICTS_EVALUATED_COUPLING_COVERAGE_H4')
                if hid=='H4' and row['structurally_coupled_scored_equation_count'] and re.search(
                        r'\bno\b[^.!?。！？]{0,180}\bstructural(?:ly)?\s+coupl(?:ed|ing)\b'
                        r'[^.!?。！？]{0,100}\bevaluated\b',sentence,re.I):
                    raise ValueError('FEEDBACK_CONTRADICTS_EVALUATED_COUPLING_COVERAGE_H4')
    return dict(protocol=PROTOCOL,status='VALIDATED_STRUCTURED_FACTS_AND_UPDATE_CONTRACTS',
        checked_facts=len(value['fact_checks']),factual_values_inserted_by_code=True,
        free_text_semantic_entailment_proved=False,
        limit='Reference existence and exact copied values are checked. Interpretation and scientific correspondence remain reviewed model judgments, not logical or physical proof.')


def defer_uncontracted_parent_revisions(answers, payload, errors):
    """Preserve accepted state when two model attempts omit all revision contracts.

    This is a control-flow fallback, not acceptance or repair of model science.
    The raw proposed states remain auditable and unresolved, preventing a false
    stable-cycle count. Other grounding failures are never handled here.
    """
    if len(answers)!=2 or errors!=['FEEDBACK_UPDATE_CONTRACT_COVERAGE_REQUIRED']*2:
        return None
    before={h['hypothesis_id']:h for h in payload['hypotheses_before']}
    if set(before)!={'H1','H2','H3','H4'} or len(payload['hypotheses_before'])!=4:
        return None
    for answer in answers:
        states=answer.get('hypotheses',[])
        if answer.get('parent_update_contracts')!=[] or len(states)!=4 or \
                {h.get('hypothesis_id') for h in states}!=set(before):
            return None
    latest={h['hypothesis_id']:h for h in answers[-1]['hypotheses']}
    pending=[]
    for hid in sorted(before):
        old,new=before[hid],latest[hid]
        if (new['claim'],new['status'])==(old['claim'],old['status']):
            continue
        pending.append(dict(hypothesis_id=hid,accepted_before=copy.deepcopy(old),
            proposed_after=copy.deepcopy(new),audit=dict(change_kind='UNCERTAIN',
                reason='Both scientific-feedback attempts omitted the required parent update contracts; no revised claim or status was accepted.',
                validation_errors=list(errors))))
    if not pending:
        return None
    return dict(round_summary='Numerical evaluation was recorded; proposed scientific revisions are deferred pending grounded review.',
        diagnosis=[],optimization_plan=[],hypotheses=copy.deepcopy(payload['hypotheses_before']),
        new_hypotheses=[],unresolved_questions=[
            'Review the uncontracted parent scientific revisions against recorded evidence before acceptance.'],
        why_expand_or_stop='Scientific adjudication is pending; this cycle cannot count toward convergence.',
        fact_checks=[],analyses=[],parent_update_contracts=[],executable_actions=[],future_experiments=[],
        _deferred_parent_updates=pending,_feedback_mode='STATE_PRESERVING_DEFERRED_UPDATE')


def audited_feedback(call, payload, schema, folder):
    """Bounded contract corrections, never requests for a favorable verdict."""
    folder=Path(folder)
    errors=[];details=[];routing=[];attempt_answers=[]
    payload=dict(payload,reference_catalogue=reference_catalogue(payload))
    schema=bind_fact_check_schema(schema,payload)
    answer=call('SCIENTIFIC_DIAGNOSTICIAN_AND_HYPOTHESIS_REVISER',payload,schema,folder/'initial')
    for attempt in range(3):
        attempt_answers.append(copy.deepcopy(answer))
        canonical,changes=normalize_references(answer,payload)
        canonical,contract_changes=complete_unchanged_claim_contracts(canonical,payload)
        changes.extend(contract_changes)
        canonical,narrative_changes=remove_retyped_narrative_literals(canonical,payload)
        changes.extend(narrative_changes)
        routing.append(dict(attempt=attempt,changes=changes))
        try:
            receipt=validate_feedback(canonical,payload)
            save(folder/'accepted_raw_feedback.json',answer)
            save(folder/'accepted_canonical_feedback.json',canonical)
            save(folder/'validation.json',dict(receipt,clarification_requested=bool(attempt),errors=errors,
                validation_details=details,reference_routing=routing,
                prior_clarification_attempt_archived=(folder/'clarification/response.json').exists(),
                scientific_validation_waived=False,raw_response_preserved=True))
            return canonical
        except ValueError as exc:
            errors.append(str(exc))
            details.append(getattr(exc,'details',dict(code=str(exc),field_path='',invalid_ids=[])))
            # A second distinct validation error can surface only after the
            # first has been corrected. Never retry a repeated error, and
            # never change scientific claims in code to obtain a pass.
            if attempt<2 and (attempt==0 or errors[-1]!=errors[-2]):
                required_ids=sorted(h['hypothesis_id'] for h in payload['hypotheses_before'])
                supplied_ids=[c['hypothesis_id'] for c in answer['parent_update_contracts']]
                previous={h['hypothesis_id']:h for h in payload['hypotheses_before']}
                proposed={h['hypothesis_id']:h for h in answer['hypotheses']}
                declared={c['hypothesis_id']:c['update_kind']
                    for c in answer['parent_update_contracts']}
                claim_contracts={hid:dict(claim_changed=(
                    proposed[hid]['claim']!=previous[hid]['claim']),
                    declared_update_kind=declared.get(hid),
                    permitted_if_claim_changed='CLAIM_REVISION',
                    permitted_if_claim_unchanged='EVIDENCE_REFRESH')
                    for hid in required_ids if hid in proposed}
                coverage={f['value']['hypothesis_id']:dict(
                    fact_id=f['fact_id'],
                    scored_complete_equation_count=f['value']['scored_complete_equation_count'],
                    structurally_coupled_scored_equation_count=f['value'].get('structurally_coupled_scored_equation_count',0))
                    for f in payload['evidence_snapshot']['facts']
                    if f['kind']=='PARENT_QUESTION_COVERAGE'}
                answer=call('SCIENTIFIC_FEEDBACK_FACT_CLARIFIER',dict(payload,
                    rejected_feedback=answer,deterministic_validation_errors=errors,
                    deterministic_validation_details=details,
                    correction_constraints=dict(required_parent_contract_ids=required_ids,
                        missing_parent_contract_ids=sorted(set(required_ids)-set(supplied_ids)),
                        scored_parent_coverage=coverage,
                        claim_contract_consistency=claim_contracts),
                    instruction='Correct the specified factual/reference/update-contract errors; supply exactly one parent_update_contract for EACH required parent ID, including any missing ID. If a parent claim changed, choose CLAIM_REVISION with a new observable prediction or restore the previous claim verbatim; EVIDENCE_REFRESH must preserve the old claim verbatim and have an empty new prediction. Do not force support, novelty or stopping. Do not claim that an H4 coupled equation was never evaluated when scored_parent_coverage reports coupled evaluations; evaluation does not itself establish the physical mechanism. F_ facts belong to fact_ids; supplied extractions belong to literature_ids. Unknown IDs must not be invented or accepted. fact_checks is exact computed evidence only. No formulas/scores in narrative.'),
                    schema,folder/('clarification' if attempt==0 else 'clarification_2'))
            else:
                break
    deferred=defer_uncontracted_parent_revisions(attempt_answers,payload,errors)
    if deferred is not None:
        save(folder/'accepted_raw_feedback.json',dict(
            status='NO_MODEL_FEEDBACK_ACCEPTED',raw_responses_archived_in=['initial/response.json','clarification/response.json']))
        save(folder/'accepted_canonical_feedback.json',deferred)
        save(folder/'deferred_scientific_update.json',dict(
            status='PENDING_SCIENTIFIC_ADJUDICATION',accepted_parent_states=payload['hypotheses_before'],
            proposed_parent_updates=deferred['_deferred_parent_updates'],
            raw_response_sha256=[fingerprint(a) for a in attempt_answers],
            validation_errors=errors,model_scientific_content_accepted=False))
        save(folder/'validation.json',dict(protocol=PROTOCOL,status='DEFERRED_UNCONTRACTED_PARENT_REVISIONS',
            errors=errors,validation_details=details,reference_routing=routing,
            clarification_requested=True,scientific_validation_waived=False,
            accepted_scientific_update=False,raw_model_answer_unaccepted=True,
            pending_parent_updates=len(deferred['_deferred_parent_updates'])))
        return deferred
    save(folder/'validation.json',dict(protocol=PROTOCOL,status='REJECTED_UNGROUNDED_FEEDBACK',
        errors=errors,validation_details=details,reference_routing=routing,
        clarification_requested=True,scientific_validation_waived=False))
    raise RuntimeError('SCIENTIFIC_FEEDBACK_FAILED_GROUNDING_AFTER_ONE_CLARIFICATION')


def quarantine_rejected_interpretation(payload, folder):
    """Keep numerical work while rejecting an ungrounded model narrative.

    This is not a replacement scientific review. No new hypothesis, claim or
    mathematical instruction is accepted, and the round cannot count as stable.
    Transport/schema failures do not reach this function.
    """
    folder = Path(folder)
    validation_path = folder/'validation.json'
    if not validation_path.exists() or read(validation_path).get('status') != 'REJECTED_UNGROUNDED_FEEDBACK':
        raise RuntimeError('SCIENTIFIC_FEEDBACK_QUARANTINE_REQUIRES_RECORDED_GROUNDING_REJECTION')
    rejected = read(validation_path)
    prior = copy.deepcopy(payload['hypotheses_before'])
    value = dict(
        round_summary='Numerical evaluation completed; model scientific interpretation was rejected by the evidence gate.',
        diagnosis=[], optimization_plan=[], hypotheses=prior, new_hypotheses=[],
        unresolved_questions=['Scientific interpretation of this evaluated batch remains pending.'],
        why_expand_or_stop='Continue only with already declared questions and independently checked mathematical proposals; this cycle cannot establish convergence.',
        fact_checks=[], analyses=[], parent_update_contracts=[],
        executable_actions=[], future_experiments=[],
        _feedback_mode='REJECTED_INTERPRETATION_QUARANTINED',
        _scientific_feedback_pending=True,
    )
    save(folder/'accepted_raw_feedback.json',dict(
        status='NO_MODEL_SCIENTIFIC_CONTENT_ACCEPTED',
        raw_responses_archived_in=['initial/response.json','clarification/response.json','clarification_2/response.json'],
        rejection_codes=rejected.get('errors',[])))
    save(folder/'accepted_canonical_feedback.json',value)
    save(folder/'scientific_feedback_quarantine.json',dict(
        status='PENDING_SCIENTIFIC_INTERPRETATION',
        model_scientific_content_accepted=False,
        accepted_parent_states=prior,
        rejection_codes=rejected.get('errors',[]),
        numerical_evaluation_preserved=True,
        convergence_cycle_allowed=False))
    return value


def verified_observations(feedback, snapshot):
    facts={f['fact_id']:f for f in snapshot['facts']}
    refs={r for a in feedback['analyses'] for r in a['fact_ids']}
    refs.update(r for a in feedback['parent_update_contracts'] for r in a['fact_ids'])
    return [facts[r] for r in sorted(refs)]


def verified_literature(feedback, evidence):
    sources={e['evidence_id']:e for e in evidence}
    refs={r for group in ('analyses','parent_update_contracts','executable_actions')
          for a in feedback[group] for r in a.get('literature_ids',[])}
    return [sources[r] for r in sorted(refs)]
