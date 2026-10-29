"""Frozen historical integration and metrics, now local to V6."""
import numpy as np
from . import engineering_state as state
from .storage import fingerprint

CASES = ("linhai", "jinhai", "pool", "ksc")

class Integrator:
    def __init__(self,data,law,mapping):
        self.data,self.law,self.mapping = data,law,mapping
        self.reference = .5*float(np.exp(law.predict_log([25.],[mapping['Cbg']],[500.]))[0])
        if not np.isfinite(self.reference) or self.reference<=0:
            raise ValueError('INVALID_REFERENCE_CURRENT')
        self.cache = {}

    def basis(self,case,c,quadrature=16):
        key = (case,fingerprint(c),quadrature)
        if key in self.cache:
            return self.cache[key]
        d = self.data[case]
        q = np.clip((d['RH_pct']-55.)/45.,0,1)**.3
        C = self.mapping['Cbg']+(self.mapping['Ccap']-self.mapping['Cbg'])*d['chloride_deposition']/(d['chloride_deposition']+20000.)
        H = np.clip(1000*q,50,1000)
        current = np.exp(self.law.predict_log(d['T_C'],C,H))
        if not np.isfinite(current).all():
            raise ValueError('CURRENT_OVERFLOW_NO_CLIPPING_APPLIED')
        start,step = state.state_clock(q*current/self.reference,d['duration_days'])
        nodes,weights = np.polynomial.legendre.leggauss(quadrature)
        u = start[:,None]+step[:,None]*(.5+.5*nodes)
        average = np.einsum('nqk,q->nk',state.state_functions(u,c),.5*weights)
        base = self.mapping['Gamma']*q*current*d['duration_days']*86400.
        increments = base[:,None]*average
        result = np.column_stack([np.bincount(d['observation_index'].astype(int),weights=increments[:,j],
                    minlength=len(d['months'])).cumsum() for j in range(c['k'])])
        self.cache[key] = result
        return result


def metrics(y,p):
    y,p = np.asarray(y),np.asarray(p)
    dy,dp = (y-y[0])[1:],(p-p[0])[1:]
    denominator = float(np.sum((dy-dy.mean())**2))
    return {'increment_r2':float(1-np.sum((dp-dy)**2)/denominator) if denominator>0 else None,
            'actual_final_depth_um':float(y[-1]),'predicted_final_depth_um':float(p[-1]),
            'endpoint_error_pct':float(100*(p[-1]/y[-1]-1)),
            'increment_scored_points':len(dy)}

