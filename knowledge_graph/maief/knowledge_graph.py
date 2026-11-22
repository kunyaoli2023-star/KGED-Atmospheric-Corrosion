"""Run-local, append-only scientific symbol memory with conservative identity."""
import re
from pathlib import Path
from .storage import read, save, fingerprint
from .symbols import REGISTRY


def normalized(text):
    return re.sub(r'\s+', ' ', text.strip())


class KnowledgeGraph:
    def __init__(self, root):
        self.path = Path(root)/'knowledge_graph.json'
        self.data = read(self.path) if self.path.exists() else {
            'version': 'V6_DYNAMIC_KNOWLEDGE_1', 'nodes': {}, 'edges': [],
            'identity_policy': 'Exact canonical mathematical role AND matching meaning/domain/context; no fuzzy semantic merge.',
            'parameter_values_shared': False}
        for name, spec in REGISTRY['variables'].items():
            self.node('coordinate:'+name, 'coordinate', canonical_symbol=name, definition=spec)
        for hid in ('H1','H2','H3','H4'):
            self.node('question:'+hid, 'question', hypothesis_id=hid)
        from .operator_library import CATALOG, VERSION, EPSILON
        for name, spec in CATALOG.items():
            self.node('operator:'+VERSION+':'+name, 'domain_operator', name=name,
                      definition=spec, epsilon=EPSILON)

    def node(self, key, kind, **fields):
        self.data['nodes'].setdefault(key, {'id':key, 'kind':kind, **fields})
        return key

    def edge(self, source, relation, target):
        edge = {'source':source,'relation':relation,'target':target}
        if edge not in self.data['edges']:
            self.data['edges'].append(edge)

    def commit(self):
        save(self.path, self.data)

    def query(self, hid):
        symbols = [n for n in self.data['nodes'].values()
                   if n['kind'] in ('parameter_concept','intermediate_concept') and n['hypothesis_id']==hid]
        failures = [n for n in self.data['nodes'].values() if n['kind']=='proposal'
                    and n['hypothesis_id']==hid and n['status'].startswith('QUARANTINED')]
        return {'version':self.data['version'], 'graph_sha256':fingerprint(self.data),
                'coordinates':REGISTRY['variables'], 'reusable_symbols':symbols,
                'recent_failures':failures[-12:],
                'reusable_constructions':[n for n in self.data['nodes'].values()
                    if n['kind']=='proposal' and n['hypothesis_id']==hid
                    and n['status'] in ('ADMITTED_FOR_NUMERICAL_COMPETITION','EXACT_DUPLICATE')
                    and n.get('construction_graph')][-12:],
                'domain_operators':[n for n in self.data['nodes'].values() if n['kind']=='domain_operator'],
                'rules':['Reuse a canonical_symbol only when its definition, role, meaning and conditions match.',
                         'Copy the listed parameter meaning/domain when adopting that concept; keep numerical coefficients independent per candidate.',
                         'New meanings or definitions create separate concepts. Same spelling is not identity.',
                         'Intermediate symbols remain explicitly declared in symbol_definitions; graph lookup does not bypass compilation.']}

    def register(self, record, protocol, module_id=None):
        from .construction import canonical_parameter_order
        p, ch = record['proposal'], record.get('checks', {})
        # Content-addressed records make recovery idempotent, including repaired versions.
        pid = 'proposal:'+fingerprint(record)
        self.node(pid, 'proposal', proposal_id=p['proposal_id'], hypothesis_id=p['hypothesis_id'],
                  status=record['admission'], expression=p['expression'], errors=ch.get('errors', []),
                  construction_graph=p.get('interface_receipt',{}).get('construction_graph'))
        self.edge('question:'+p['hypothesis_id'], 'has_proposal', pid)
        receipt=p.get('interface_receipt',{})
        for item in receipt.get('regularization',[]):
            self.edge(pid, 'uses_regularized_operator',
                      'operator:'+receipt['operator_library']+':'+item['operator'])
        if record['admission'] not in ('ADMITTED_FOR_NUMERICAL_COMPETITION','EXACT_DUPLICATE'):
            self.commit()
            return {}
        if module_id:
            self.node('module:'+module_id, 'module', module_id=module_id)
            self.edge(pid, 'implements', 'module:'+module_id)
        for link in p.get('source_links', []):
            eid = self.node('evidence:'+link['evidence_id'], 'evidence_reference', evidence_id=link['evidence_id'])
            self.edge(pid, 'cites', eid)
        compiled = ch.get('compiled_proposal', p)
        order = canonical_parameter_order(compiled, protocol)
        params = {x['name']:x for x in compiled['parameters']}
        context = {'mathematics':ch['dedup_key'], 'domains':protocol['domains'], 'reference':protocol['reference']}
        mapping = {}
        for index, name in enumerate(order):
            param = params[name]
            signature = {**context, 'role_index':index, 'meaning':normalized(param['meaning']),
                         'dimension':param['dimension'], 'domain':param['domain']}
            key = 'parameter:'+fingerprint(signature)
            canonical = 'p_'+fingerprint(signature)[:12]
            self.node(key, 'parameter_concept', canonical_symbol=canonical,
                      hypothesis_id=p['hypothesis_id'], definition=signature, aliases=[])
            node = self.data['nodes'][key]
            if name not in node['aliases']:node['aliases'].append(name)
            self.edge(pid,'uses_parameter_concept',key)
            mapping[name] = canonical
        # Intermediate identity uses expanded algebra with stable parameter concept names.
        from .construction import symbolic
        from .symbols import variable_symbols
        import sympy as sp
        symbols = variable_symbols()
        symbols.update({name:sp.Symbol(canonical) for name,canonical in mapping.items()})
        declared = {d['name']:d for d in p.get('symbol_definitions', [])}
        for d in ch.get('symbol_compilation',{}).get('definitions', []):
            signature = {**context, 'expression':sp.srepr(sp.simplify(symbolic(d['expanded'],symbols))),
                         'meaning':normalized(declared[d['symbol']]['meaning'])}
            key = 'intermediate:'+fingerprint(signature)
            canonical = 's_'+fingerprint(signature)[:12]
            self.node(key,'intermediate_concept',canonical_symbol=canonical,
                      hypothesis_id=p['hypothesis_id'],definition=signature,aliases=[],
                      expanded_expression=d['expanded'],parameter_symbols=mapping.copy())
            node=self.data['nodes'][key]
            if d['symbol'] not in node['aliases']:node['aliases'].append(d['symbol'])
            self.edge(pid,'defines',key)
            mapping[d['symbol']]=canonical
        # Receipt records a canonical display vocabulary, never ties fitted coefficients.
        self.data['nodes'][pid]['canonical_symbol_mapping']=mapping
        self.commit()
        return {'graph_proposal_id':pid,'canonical_symbols':mapping,'fit_parameters_shared':False}

    def evaluations(self, records):
        for record in records:
            eid='evaluation:'+fingerprint(record)
            self.node(eid,'evaluation',equation_key=record['equation_key'],score=record.get('score'),
                      error=record.get('error',record.get('error_class')))
            for mid in record.get('modules',[]):
                self.node('module:'+mid,'module',module_id=mid)
                self.edge(eid,'evaluates','module:'+mid)
        self.commit()
