#!/usr/bin/env python3
"""Train-to-validation ridge probe for state-conditioned decomposition calibration."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_provider.data_factory import data_provider
from models.GraphMamba import Model
from scripts.diagnose_graphmamba_representations import command_args, locate_record

OUTPUT = ROOT / "logs" / "graphmamba_state_conditioned_stc"
TASKS = (("ETTm1", 96), ("ETTm2", 96), ("ETTm1", 720), ("ETTm2", 720))
N_STATE = 4
N_FEATURES = 2 * N_STATE


def forward_parts(model, x):
    means=x.mean(1,keepdim=True); centered=x-means
    stdev=torch.sqrt(torch.var(centered,dim=1,keepdim=True,unbiased=False)+1e-5)
    normalized=centered/stdev
    seasonal_input,trend_input=model.decomposition(normalized)
    trend=model.trend_projection(trend_input.permute(0,2,1)).permute(0,2,1)*stdev
    seasonal_input=seasonal_input.permute(0,2,1)
    tokens=torch.cat((model.long_patch_embedding(seasonal_input)+model.variable_embedding,
                      model.short_patch_embedding(seasonal_input)+model.variable_embedding),dim=-1)
    seasonal=model.head(model.encoder(tokens)+model.graph_mixer(tokens))*stdev
    return seasonal,trend,means,normalized


def state_features(normalized):
    # [B,L,V] -> [B,V,4]. Every feature uses encoder history only.
    length=normalized.shape[1]; half=length//2; recent=normalized[:,half:]
    time=torch.linspace(-1,1,recent.shape[1],device=normalized.device)
    slope=(recent*time[None,:,None]).mean(1)
    diff_std=(recent[:,1:]-recent[:,:-1]).std(1,unbiased=False)
    level_shift=recent.mean(1)-normalized[:,:half].mean(1)
    raw=torch.stack((slope,diff_std,level_shift),dim=-1)
    # Bounded features avoid extrapolation explosions under temporal shift.
    return torch.cat((torch.ones_like(raw[...,:1]),torch.tanh(raw)),dim=-1)


def accumulate(model,loader,pred_len,n_vars,ridge=False):
    xtx=torch.zeros((n_vars,N_FEATURES,N_FEATURES),dtype=torch.float64)
    xtr=torch.zeros((n_vars,N_FEATURES),dtype=torch.float64)
    for batch_x,batch_y,*_ in loader:
        x=batch_x.float().cuda(); y=batch_y[:,-pred_len:,:].float().cuda()
        with torch.no_grad():
            seasonal,trend,means,normalized=forward_parts(model,x)
            state=state_features(normalized)
            design=torch.cat((seasonal[:,:,:,None]*state[:,None,:,:],
                              trend[:,:,:,None]*state[:,None,:,:]),dim=-1)
            residual=y-(seasonal+trend+means)
        design=design.double().cpu(); residual=residual.double().cpu()
        xtx+=torch.einsum('btvi,btvj->vij',design,design)
        xtr+=torch.einsum('btvi,btv->vi',design,residual)
    if ridge:
        # Fixed scale-relative ridge; intercept-like terms are also regularized.
        scale=xtx.diagonal(dim1=-2,dim2=-1).mean(-1).clamp_min(1e-12)
        xtx=xtx+1e-3*scale[:,None,None]*torch.eye(N_FEATURES,dtype=torch.float64)[None]
    return torch.linalg.solve(xtx+1e-10*torch.eye(N_FEATURES,dtype=torch.float64)[None],xtr.unsqueeze(-1)).squeeze(-1)


def evaluate(model,loader,pred_len,coefficients):
    coefficients=coefficients.cuda().float(); sums={k:0.0 for k in ('sse','sae','csse','csae')}; count=0
    with torch.no_grad():
        for batch_x,batch_y,*_ in loader:
            x=batch_x.float().cuda(); y=batch_y[:,-pred_len:,:].float().cuda()
            seasonal,trend,means,normalized=forward_parts(model,x); state=state_features(normalized)
            seasonal_delta=torch.einsum('bvk,vk->bv',state,coefficients[:,:N_STATE])
            trend_delta=torch.einsum('bvk,vk->bv',state,coefficients[:,N_STATE:])
            # Bound sample-specific final multipliers to [0,2].
            seasonal_delta=seasonal_delta.clamp(-1,1); trend_delta=trend_delta.clamp(-1,1)
            baseline=seasonal+trend+means
            corrected=baseline+seasonal*seasonal_delta[:,None,:]+trend*trend_delta[:,None,:]
            error=y-baseline; corrected_error=y-corrected
            sums['sse']+=error.square().sum().item();sums['sae']+=error.abs().sum().item()
            sums['csse']+=corrected_error.square().sum().item();sums['csae']+=corrected_error.abs().sum().item();count+=y.numel()
    return {'base_mse':sums['sse']/count,'base_mae':sums['sae']/count,
            'corrected_mse':sums['csse']/count,'corrected_mae':sums['csae']/count,
            'mse_improvement_pct':100*(sums['sse']-sums['csse'])/sums['sse'],
            'mae_improvement_pct':100*(sums['sae']-sums['csae'])/sums['sae']}


def diagnose(dataset,pred_len):
    record,checkpoint=locate_record(dataset,pred_len);args=command_args(record['command'])
    torch.manual_seed(2021);model=Model(args).cuda().eval();model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True))
    _,train_loader=data_provider(args,'train');_,val_loader=data_provider(args,'val')
    coefficients=accumulate(model,train_loader,pred_len,args.enc_in,ridge=True)
    metrics=evaluate(model,val_loader,pred_len,coefficients)
    return {'dataset':dataset,'pred_len':pred_len,**metrics,'coefficients':coefficients.tolist(),
            'state_features':['constant','recent_slope','difference_std','half_mean_shift'],
            'fit_split':'train','evaluation_split':'val','test_accessed':False,'checkpoint':str(checkpoint)}


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True);rows=[]
    for dataset,pred_len in TASKS:
        print(f'State-conditioned STC: {dataset}-{pred_len}',flush=True);row=diagnose(dataset,pred_len);rows.append(row)
        print(json.dumps(row,sort_keys=True),flush=True);(OUTPUT/f'{dataset.lower()}_{pred_len}.json').write_text(json.dumps(row,indent=2)+'\n')
    with (OUTPUT/'summary.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
