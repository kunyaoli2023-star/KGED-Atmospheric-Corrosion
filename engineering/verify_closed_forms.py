"""Independent closed-form reconstruction of the two saved laboratory laws."""
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
frozen=read(root/'PREDICTIONS_FROZEN.json')
lab=read(root.parent/'data/data.json')
t=(np.asarray(lab['T_C'])+273.15)/298.15
c=np.asarray(lab['C_wt']);h=np.asarray(lab['h_um'])/200
checks={}
for alias in ('parsimonious','best_accuracy'):
    r=read(root/f'{alias}_equation.json')
    p={m['hypothesis']:m['parameters'] for m in r['physical_modules']}
    B,D=p['H2']['p0'],p['H2']['p1']
    salt=np.log((1+B*c*np.exp(-D*c))/(1+B*3.5*np.exp(-D*3.5)))
    if alias=='parsimonious':
        a=p['H3']['p0']
        pred=r['intercept']+p['H1']['p0']*(1-1/t)+salt+np.log(h*(a+1)/(a+h*h))
    else:
        A,K=p['H1']['p0'],p['H1']['p1']
        E,n,L=p['H3']['p0'],p['H3']['p1'],p['H3']['p2']
        pred=r['intercept']+A*(np.tanh(K*t)-np.tanh(K))+salt+np.log((1+E*h**n*np.exp(-L*h))/(1+E*np.exp(-L)))
    error=float(np.max(np.abs(pred-np.asarray(r['full_predictions']))))
    np.testing.assert_allclose(pred,r['full_predictions'],atol=2e-11,rtol=0)
    checks[alias]={'independent_closed_form_max_log_difference':error,'i_reference':float(np.exp(r['intercept']))}
(root/'CLOSED_FORM_VERIFICATION.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(json.dumps(checks,indent=2))
