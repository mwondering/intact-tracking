"""Check whether a nonlinear full-latent readout reveals omitted DR coordinates."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from intact_tracking.router_information_probe import regression_metrics
from probe_moe_latent_payload import load_case


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--root',required=True)
    args=p.parse_args();source,root=Path(args.source),Path(args.root);torch.set_num_threads(4)
    a=load_case(source,'sample',30402);b=load_case(root,'heldout_sample',30403)
    schema=json.loads((source/'sample_seed30402/result.json').read_text())['dr_schema']
    lower=np.asarray(schema['lower']);width=np.asarray(schema['upper'])-lower
    order=np.random.default_rng(887).permutation(len(a['raw_dr']));fit=np.isin(a['worlds'],order[:768]);val=~fit
    y=(a['raw_dr'][a['worlds']]-lower)/width;test_y=(b['raw_dr'][b['worlds']]-lower)/width
    x=a['latent'];xm,xs=x[fit].mean(0),x[fit].std(0).clip(1e-6);ym,ys=y[fit].mean(0),y[fit].std(0).clip(1e-6)
    train_x=torch.from_numpy(((x[fit]-xm)/xs).astype(np.float32));train_y=torch.from_numpy(((y[fit]-ym)/ys).astype(np.float32))
    val_x=torch.from_numpy(((x[val]-xm)/xs).astype(np.float32));test_x=torch.from_numpy(((b['latent']-xm)/xs).astype(np.float32))
    weight=np.zeros(67,dtype=np.float32)
    for cols in schema['groups'].values():weight[cols]=1/(6*len(cols))
    weight=torch.from_numpy(weight);rows=[];predictions=[]
    for seed in (571,572,573):
        torch.manual_seed(seed);model=nn.Sequential(nn.Linear(64,256),nn.SiLU(),nn.Linear(256,128),nn.SiLU(),nn.Linear(128,67))
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3)
        best=-np.inf;best_epoch=0;best_state=None;history=[]
        for epoch in range(1,151):
            model.train()
            for ids in torch.randperm(len(train_x)).split(1024):
                optimizer.zero_grad(set_to_none=True)
                (((model(train_x[ids])-train_y[ids]).square()*weight).sum(-1).mean()).backward();optimizer.step()
            if epoch%5==0 or epoch==1:
                model.eval()
                with torch.inference_mode():prediction=model(val_x).numpy()*ys+ym
                metrics=regression_metrics(y[val],prediction,schema['groups']);score=float(np.mean(list(metrics['groups_r2'].values())))
                history.append({'epoch':epoch,'six_group_mean_r2':score})
                if score>best:best,best_epoch,best_state=score,epoch,copy.deepcopy(model.state_dict())
                if epoch>=30 and epoch-best_epoch>=25:break
        model.load_state_dict(best_state);model.eval()
        with torch.inference_mode():prediction=model(test_x).numpy()*ys+ym
        predictions.append(prediction)
        row={'initialization_seed':seed,'best_epoch':best_epoch,'validation_six_group_mean_r2':best,
             'heldout_metrics':regression_metrics(test_y,prediction,schema['groups']),'history':history};rows.append(row)
        print(json.dumps({'seed':seed,'epoch':best_epoch,'groups':row['heldout_metrics']['groups_r2']}),flush=True)
    report={'protocol':'Full64-D latent to67 normalized DR coordinates,64-256-128-67 SiLU MLP; six DR families equal loss weight. Fit768 and validate256 whole worlds from seed30402. Test independent seed30403. Three decoder initializations, not three policy training seeds. This check does not alter any selected router.',
            'limit':'Poor readout generalization does not prove mathematical absence of information.',
            'rows':rows,'ensemble_heldout_metrics':regression_metrics(test_y,np.mean(predictions,axis=0),schema['groups'])}
    (root/'remaining_dr_nonlinear_probe.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
