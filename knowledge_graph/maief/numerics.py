"""Fit independently generated modules with their saved parameter/domain contracts.

No archived equation, coefficients or scores. Analytic parameter Jacobians and
fold-local multistarts; no clipping of predictions or repair of invalid algebra.
"""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import sympy as sp
from scipy.optimize import least_squares

from . import construction

from .symbols import VARIABLES as VARS, REFERENCE as REF, variable_symbols, coordinates, validate_inputs
_MODULE_CACHE={};_INPUT_CACHE={}


def save(path,value):construction.save(path,value)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(ids):return hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest()


class Module:
    def __init__(self,record):
        self.record=record;self.id=record['module_id'];self.p=record['proposal']
        self.params=self.p['parameters'];self.n=len(self.params)
        self.symbols=variable_symbols()
        self.symbols.update({p['name']:sp.Symbol(p['name'],**({'positive':True} if p['domain']=='POSITIVE' else {'real':True})) for p in self.params})
        self.expr=construction.symbolic(self.p['expression'],self.symbols)
        self.args=[self.symbols[v] for v in VARS]+[self.symbols[p['name']] for p in self.params]
        self.fn=sp.lambdify(self.args,self.expr,'numpy',cse=True)
        self.partials=[sp.diff(self.expr,self.symbols[p['name']]) for p in self.params]
        self.jacs=[sp.lambdify(self.args,x,'numpy',cse=True) for x in self.partials]
        self.zero_limits={}
        self.conditions=[]
        for condition in self.p['conditions']:
            expr=construction.symbolic(condition['expression'],self.symbols)
            derivatives=[sp.diff(expr,self.symbols[p['name']]) for p in self.params]
            self.conditions.append((condition,expr,sp.lambdify(self.args,expr,'numpy',cse=True),
                [sp.lambdify(self.args,x,'numpy',cse=True) for x in derivatives]))
        self.nodes=sum(1 for _ in ast.walk(ast.parse(self.p['expression'],mode='eval')))

    def physical(self,z):
        return np.asarray([np.exp(x) if p['domain']=='POSITIVE' else x for p,x in zip(self.params,z)])

    def arguments(self,values,z):
        theta=self.physical(z)
        return [values[v] for v in VARS]+list(theta),theta

    def response(self,values,z,jac=False):
        args,theta=self.arguments(values,z);n=len(np.atleast_1d(values['tau']))
        reference={v:np.asarray([REF[v]]) for v in VARS}
        refargs,_=self.arguments(reference,z)
        with np.errstate(all='ignore'):
            y=np.broadcast_to(np.asarray(self.fn(*args),float),(n,))-float(np.asarray(self.fn(*refargs)).reshape(-1)[0])
        if not np.isfinite(y).all():raise ValueError('Nonfinite centred response: '+self.id)
        if not jac:return y
        columns=[]
        for j,(partial,fn,p) in enumerate(zip(self.partials,self.jacs,self.params)):
            with np.errstate(all='ignore'):
                col=np.broadcast_to(np.asarray(fn(*args),float),(n,)).copy()
                ref=float(np.asarray(fn(*refargs)).reshape(-1)[0])
            bad=~np.isfinite(col)
            if bad.any():
                # A removable c=0 parameter-derivative singularity is evaluated
                # by its symbolic one-sided limit, never by arbitrary clipping.
                if np.any(np.asarray(values['c'])[bad]!=0):raise ValueError('Nonfinite parameter derivative')
                if j not in self.zero_limits:
                    limit=sp.limit(partial,self.symbols['c'],0,dir='+')
                    self.zero_limits[j]=sp.lambdify(self.args,limit,'numpy',cse=True)
                replacement=np.broadcast_to(np.asarray(self.zero_limits[j](*args),float),(n,))
                col[bad]=replacement[bad]
            col-=ref
            if p['domain']=='POSITIVE':col*=theta[j]
            if not np.isfinite(col).all():raise ValueError('Unresolved parameter derivative')
            columns.append(col)
        return y,np.asarray(columns).T if columns else np.empty((n,0))

    def constraints(self,values,z,jac=False):
        args,theta=self.arguments(values,z);res=[];columns=[]
        n=len(np.atleast_1d(values['tau']))
        for condition,expr,fn,jacs in self.conditions:
            with np.errstate(all='ignore'):v=np.atleast_1d(np.asarray(fn(*args),float))
            if v.size not in (1,n) or not np.isfinite(v).all():raise ValueError('Nonfinite domain condition')
            margin=1e-9 if condition['relation']=='GT_ZERO' else 0.
            res.extend(v-margin)
            if jac:
                matrix=[]
                for p,x,jfn in zip(self.params,theta,jacs):
                    a=np.broadcast_to(np.asarray(jfn(*args),float),v.shape).copy()
                    if p['domain']=='POSITIVE':a*=x
                    matrix.append(a)
                columns.append(np.asarray(matrix).T if matrix else np.empty((len(v),0)))
        values=np.asarray(res)
        return (values,np.concatenate(columns) if columns else np.empty((0,self.n))) if jac else values


def get_module(record):
    token=hashlib.sha256(json.dumps(record['proposal'],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    if token not in _MODULE_CACHE:_MODULE_CACHE[token]=Module(record)
    return _MODULE_CACHE[token]


class Evaluator:
    def __init__(self,inputs,options):
        inputs=Path(inputs);self.options=options
        cache=str(inputs.resolve())
        if cache not in _INPUT_CACHE:_INPUT_CACHE[cache]=(read(inputs/'data.json'),read(inputs/'folds.json'))
        self.table,self.folds=_INPUT_CACHE[cache]
        self.y=np.asarray(self.table['log_i'],float)
        self.x=coordinates(self.table)
        self.coarse=[]
        for axis in sorted({f['axis'] for f in self.folds}):
            ix=[j for j,f in enumerate(self.folds) if f['axis']==axis]
            self.coarse.extend([ix[0],ix[len(ix)//2]])
        self.coarse=sorted(set(self.coarse))
        protocol=read(inputs/'module_protocol.json')
        validate_inputs(inputs, protocol)
        axes=[np.linspace(*protocol['domains'][v],3) for v in VARS]
        grid=np.meshgrid(*axes,indexing='ij')
        self.domain={v:np.r_[a.ravel(),REF[v]] for v,a in zip(VARS,grid)}

    def fit(self,modules,indices,label,starts,max_nfev):
        npar=1+sum(m.n for m in modules)
        if npar>=len(indices):raise ValueError('No residual degrees of freedom')
        # Keep the historical default for existing runs. Later construction
        # rounds can strengthen inequality enforcement without changing their
        # scientific parameter domains or the first-round optimizer budget.
        constraint_penalty=float(self.options.get('constraint_penalty',1000.))
        if not np.isfinite(constraint_penalty) or constraint_penalty<=0:
            raise ValueError('constraint_penalty must be finite and positive')
        bounds=[(-30.,5.)]+[(-8.,6.) if p['domain']=='POSITIVE' else (-20.,20.) for m in modules for p in m.params]
        lo=np.asarray([a for a,b in bounds]);hi=np.asarray([b for a,b in bounds])
        xx={v:a[indices] for v,a in self.x.items()};yy=self.y[indices]
        slices=[];cursor=1
        for m in modules:slices.append(slice(cursor,cursor+m.n));cursor+=m.n
        seed=int(hashlib.sha256((str(self.options['seed'])+label).encode()).hexdigest()[:8],16)
        rng=np.random.default_rng(seed)
        initials=[]
        # Feasible start selection uses contracts/predictors, never validation y.
        for trial in range(120):
            z=np.zeros(npar);z[0]=np.mean(yy)
            if trial:
                for m,sl in zip(modules,slices):
                    z[sl]=[rng.uniform(-1.8,1.4) if p['domain']=='POSITIVE' else rng.uniform(-.6,.6) for p in m.params]
            try:
                if any(np.any(m.constraints(self.domain,z[sl])<0) for m,sl in zip(modules,slices)):continue
                for m,sl in zip(modules,slices):m.response(xx,z[sl],True)
            except (ValueError,TypeError,OverflowError,NotImplementedError):continue
            initials.append(z)
            if len(initials)>=starts:break
        if not initials:raise ValueError('No feasible multistart under saved module conditions')
        memo={}
        def calculate(z):
            if 'z' in memo and np.array_equal(z,memo['z']):return memo['r'],memo['j']
            pred=np.full(len(indices),z[0]);jac=np.zeros((len(indices),npar));jac[:,0]=1
            penalties=[];penalty_j=[]
            try:
                for m,sl in zip(modules,slices):
                    term,mjac=m.response(xx,z[sl],True);pred+=term;jac[:,sl]=mjac
                    c,cj=m.constraints(self.domain,z[sl],True)
                    active=c<0;penalties.extend(np.minimum(c,0)*constraint_penalty)
                    pj=np.zeros((len(c),npar));pj[:,sl]=np.where(active[:,None],cj,0.)*constraint_penalty;penalty_j.append(pj)
                r=np.r_[pred-yy,penalties]
                j=np.concatenate([jac]+penalty_j) if penalty_j else jac
                if not np.isfinite(r).all() or not np.isfinite(j).all():raise ValueError('Nonfinite objective')
            except (ValueError,OverflowError,FloatingPointError,TypeError):
                size=len(indices)+sum(len(m.constraints(self.domain,initials[0][sl])) for m,sl in zip(modules,slices))
                r=np.full(size,1e6);j=np.zeros((size,npar))
            memo.update(z=z.copy(),r=r,j=j);return r,j
        best=None;diagnostics=[]
        for initial in initials:
            memo.clear()
            try:
                fit=least_squares(lambda z:calculate(z)[0],initial,jac=lambda z:calculate(z)[1],
                    bounds=(lo,hi),max_nfev=max_nfev,x_scale='jac',ftol=1e-7,xtol=1e-7,gtol=1e-7)
                feasible=all(np.all(m.constraints(self.domain,fit.x[sl])>=-1e-8) for m,sl in zip(modules,slices))
                pred=self.predict(modules,slices,fit.x,xx)
                sse=float(np.sum((pred-yy)**2))
                diagnostics.append({'success':bool(fit.success),'nfev':int(fit.nfev),'feasible':feasible,'sse':sse})
                if fit.success and feasible and (best is None or sse<best[0]):best=(sse,fit,slices)
            except (ValueError,TypeError,FloatingPointError,OverflowError) as e:
                diagnostics.append({'success':False,'error':str(e)[:160]})
        if best is None:raise ValueError('No converged feasible multistart')
        return best[1],best[2],diagnostics

    def predict(self,modules,slices,z,xx):
        pred=np.full(len(xx['tau']),z[0])
        for m,sl in zip(modules,slices):
            if np.any(m.constraints(xx,z[sl]) < -1e-8):raise ValueError('Prediction outside module conditions')
            pred+=m.response(xx,z[sl])
        if not np.isfinite(pred).all():raise ValueError('Nonfinite complete prediction')
        return pred

    def evaluate(self,records,fidelity):
        modules=[get_module(r) for r in records];npar=1+sum(m.n for m in modules)
        fold_ids=self.coarse if fidelity=='screen' else list(range(len(self.folds)))
        starts=self.options['screen_starts'] if fidelity=='screen' else self.options['final_starts']
        max_nfev=self.options['screen_nfev'] if fidelity=='screen' else self.options['final_nfev']
        fold_rows=[];axes={}
        for j in fold_ids:
            fold=self.folds[j];tr=np.asarray(fold['train'],int);va=np.asarray(fold['validation'],int)
            if np.intersect1d(tr,va).size:raise ValueError('Fold leakage')
            fit,sections,diag=self.fit(modules,tr,'|fold='+str(j),starts,max_nfev)
            pred=self.predict(modules,sections,fit.x,{v:a[va] for v,a in self.x.items()})
            rmse=float(np.sqrt(np.mean((pred-self.y[va])**2)));axes.setdefault(fold['axis'],[]).append(rmse)
            fold_rows.append({'fold':j,'axis':fold['axis'],'train_indices':tr.tolist(),'validation_indices':va.tolist(),
                'observed':self.y[va].tolist(),'predicted':pred.tolist(),'rmse':rmse,
                'latent_parameters':fit.x.tolist(),'optimizer_diagnostics':diag})
        score={'cv_log_rmse':float(np.mean([np.mean(a) for a in axes.values()])),
            'axis_rmse':{a:float(np.mean(v)) for a,v in axes.items()},'parameter_count':npar,
            'ast_nodes':sum(m.nodes for m in modules),'validation_folds':len(fold_ids),
            'fidelity':fidelity,'independent_test_claimed':False}
        result={'modules':[r['module_id'] for r in records],'score':score,'fold_rows':fold_rows,'admissible':True}
        if fidelity=='final':
            ix=np.arange(len(self.y));fit,sections,diag=self.fit(modules,ix,'|full',max(starts,6),max(800,max_nfev))
            pred=self.predict(modules,sections,fit.x,self.x)
            cols=[np.ones(len(self.y))]
            physical=[]
            for m,sl in zip(modules,sections):
                _,jac=m.response(self.x,fit.x[sl],True);cols.extend(jac.T)
                theta=m.physical(fit.x[sl]);physical.append({'module_id':m.id,'hypothesis':m.p['hypothesis_id'],
                    'expression':m.p['expression'],'parameters':{p['name']:float(x) for p,x in zip(m.params,theta)}})
            matrix=np.asarray(cols).T;norms=np.linalg.norm(matrix,axis=0)
            scaled=matrix/np.maximum(norms,1e-30);singular=np.linalg.svd(scaled,compute_uv=False)
            rank=int(np.sum(singular>singular[0]*1e-7));boundary=int(np.sum((fit.x-np.asarray([-30.]+[-8. if p['domain']=='POSITIVE' else -20. for m in modules for p in m.params])<1e-4)|
                (np.asarray([5.]+[6. if p['domain']=='POSITIVE' else 20. for m in modules for p in m.params])-fit.x<1e-4)))
            score.update(train_log_rmse=float(np.sqrt(np.mean((pred-self.y)**2))),
                train_log_r2=float(1-np.sum((pred-self.y)**2)/np.sum((self.y-self.y.mean())**2)),
                response_jacobian_rank=rank,boundary_count=boundary,
                scaled_condition_number=float(singular[0]/max(singular[-1],1e-30)))
            result.update(intercept=float(fit.x[0]),physical_modules=physical,full_predictions=pred.tolist(),
                full_optimizer_diagnostics=diag)
            result['admissible']=rank==npar and boundary==0
        return result


def worker(task):
    inputs,records,fidelity,options,destination=task;started=time.perf_counter()
    try:result=Evaluator(inputs,options).evaluate(records,fidelity)
    except Exception as e:
        result={'modules':[r['module_id'] for r in records],'admissible':False,
            'error_type':type(e).__name__,'error':str(e)[:240],'score':None}
    result.update(elapsed_seconds=time.perf_counter()-started,equation_key=key(result['modules']))
    save(destination,result);return result
