"""Score-only features and explicit chronological provenance for stacking."""
from __future__ import annotations

import numpy as np
from .recall_extension_metrics import percentile_scores
from .recall_weighted_metrics import aligned,check_data,earlier_months,pool


def score_matrix(items,members,representation,clip=1e-6):
    if not members or len(set(members))!=len(members) or not 0<clip<.5:
        raise ValueError("Invalid fixed score inputs")
    if representation not in ["logit","logit_percentile"]:
        raise ValueError("Unknown score representation")
    ref=items[members[0]];check_data(ref);columns=[];names=[]
    for name in members:
        data=items[name];aligned(ref,data)
        if np.any((data['scores']<0)|(data['scores']>1)):
            raise ValueError("Base scores must be in [0,1]")
        p=np.clip(data['scores'],clip,1-clip)
        columns.append(np.log(p/(1-p)));names.append(name+"__logit")
        if representation=="logit_percentile":
            columns.append(percentile_scores(data['scores']));names.append(name+"__percentile")
    return np.column_stack(columns),names,ref


def verify_source(record,month):
    if record['eval_month']!=month or record['train_month_max']>=month:
        raise ValueError("Base/combiner predictions must be out of time")
    if record.get('in_sample',False):
        raise ValueError("In-sample predictions are ineligible")


def training_matrix(history,records,spec,evaluation,clip=1e-6):
    months=list(range(1,evaluation));earlier_months(months,evaluation)
    arrays=[];populations=[];names=None
    for month in months:
        for member in spec['members']:
            verify_source(records[(member,month)],month)
        x,cols,ref=score_matrix({n:history[n][month] for n in spec['members']},spec['members'],spec['representation'],clip)
        if names is not None and cols!=names:raise ValueError("Feature order changed")
        arrays.append(x);populations.append(ref);names=cols
    return np.concatenate(arrays),names,pool(populations),months


def policy_history(history,records,name,evaluation,validation_months):
    months=[m for m in validation_months if m<evaluation];earlier_months(months,evaluation)
    for month in months:verify_source(records[(name,month)],month)
    return {m:history[name][m] for m in months}


def matched_control(spec,specs):
    if spec['group']!='stack':return None
    matches=[s for s in specs if s['group']=='single' and all(s[k]==spec[k] for k in ['representation','C','pos_weight'])]
    if len(matches)!=1:raise ValueError("Matched control is missing or ambiguous")
    return matches[0]
