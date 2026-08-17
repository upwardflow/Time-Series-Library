#!/usr/bin/env python3
"""Train-to-validation probe for shared versus variable-specific trend heads."""

from __future__ import annotations

import csv, json, sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from data_provider.data_factory import data_provider
from models.GraphMamba import Model
from scripts.diagnose_graphmamba_representations import command_args, locate_record

OUTPUT=ROOT/'logs'/'graphmamba_variable_trend_head'
TASKS=(("ETTm1",96),("ETTm2",96),("ETTm1",720),("ETTm2",720))

def parts(model,x):
    means=x.mean(1,keepdim=True);centered=x-means
    stdev=torch.sqrt(torch.var(centered,dim=1,keepdim=True,unbiased=False)+1e-5)
    seasonal_input,trend=model.decomposition(centered/stdev)
    seasonal_input=seasonal_input.permute(0,2,1)
    tokens=torch.cat((model.long_patch_embedding(seasonal_input)+model.variable_embedding,
                      model.short_patch_embedding(seasonal_input)+model.variable_embedding),dim=-1)
    seasonal=model.head(model.encoder(tokens)+model.graph_mixer(tokens))
    return seasonal,trend,means,stdev

def fit(model,loader,pred_len,n_vars):
    dim=model.seq_len+1
    per_xtx=torch.zeros((n_vars,dim,dim),dtype=torch.float64);per_xty=torch.zeros((n_vars,dim,pred_len),dtype=torch.float64)
    shared_xtx=torch.zeros((dim,dim),dtype=torch.float64);shared_xty=torch.zeros((dim,pred_len),dtype=torch.float64)
    with torch.no_grad():
        for batch_x,batch_y,*_ in loader:
            x=batch_x.float().cuda();y=batch_y[:,-pred_len:,:].float().cuda()
            seasonal,trend,means,stdev=parts(model,x)
            target=(y-means)/stdev-seasonal
            design=torch.cat((trend,torch.ones_like(trend[:,:1])),dim=1).permute(0,2,1).double().cpu()
            target=target.permute(0,2,1).double().cpu()
            per_xtx+=torch.einsum('bvi,bvj->vij',design,design);per_xty+=torch.einsum('bvi,bvp->vip',design,target)
            flat=design.reshape(-1,dim);flat_y=target.reshape(-1,pred_len)
            shared_xtx+=flat.T@flat;shared_xty+=flat.T@flat_y
    def solve(xtx,xty):
        scale=xtx.diagonal(dim1=-2,dim2=-1).mean(-1).clamp_min(1e-12)
        eye=torch.eye(dim,dtype=torch.float64)
        return torch.linalg.solve(xtx+1e-3*scale[...,None,None]*eye,xty)
    return solve(shared_xtx,shared_xty),solve(per_xtx,per_xty)

def evaluate(model,loader,pred_len,shared,per):
    shared=shared.cuda().float();per=per.cuda().float();sums={k:0. for k in ('base2','base1','shared2','shared1','per2','per1')};count=0
    with torch.no_grad():
        for batch_x,batch_y,*_ in loader:
            x=batch_x.float().cuda();y=batch_y[:,-pred_len:,:].float().cuda();seasonal,trend,means,stdev=parts(model,x)
            design=torch.cat((trend,torch.ones_like(trend[:,:1])),dim=1).permute(0,2,1)
            original=(seasonal+model.trend_projection(trend.permute(0,2,1)).permute(0,2,1))*stdev+means
            shared_trend=torch.einsum('bvi,ip->bvp',design,shared).permute(0,2,1)
            per_trend=torch.einsum('bvi,vip->bvp',design,per).permute(0,2,1)
            preds={'base':original,'shared':(seasonal+shared_trend)*stdev+means,'per':(seasonal+per_trend)*stdev+means}
            for name,pred in preds.items():
                error=y-pred;sums[name+'2']+=error.square().sum().item();sums[name+'1']+=error.abs().sum().item()
            count+=y.numel()
    out={name+'_mse':sums[name+'2']/count for name in ('base','shared','per')};out.update({name+'_mae':sums[name+'1']/count for name in ('base','shared','per')})
    out['per_vs_base_mse_pct']=100*(out['base_mse']-out['per_mse'])/out['base_mse'];out['per_vs_shared_mse_pct']=100*(out['shared_mse']-out['per_mse'])/out['shared_mse']
    return out

def diagnose(dataset,pred_len):
    record,checkpoint=locate_record(dataset,pred_len);args=command_args(record['command']);torch.manual_seed(2021)
    model=Model(args).cuda().eval();model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True))
    _,train=data_provider(args,'train');_,val=data_provider(args,'val');shared,per=fit(model,train,pred_len,args.enc_in)
    return {'dataset':dataset,'pred_len':pred_len,**evaluate(model,val,pred_len,shared,per),'fit_split':'train','evaluation_split':'val','test_accessed':False}

def main():
    OUTPUT.mkdir(parents=True,exist_ok=True);rows=[]
    for dataset,pred_len in TASKS:
        print(f'Variable trend head: {dataset}-{pred_len}',flush=True);row=diagnose(dataset,pred_len);rows.append(row);print(json.dumps(row,sort_keys=True),flush=True)
    with (OUTPUT/'summary.csv').open('w',newline='',encoding='utf-8') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return 0
if __name__=='__main__':raise SystemExit(main())
