#!/usr/bin/env python3
"""Condition old memory on recent history before correcting frozen GraphMamba."""

from __future__ import annotations
import copy,csv,json,sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from data_provider.data_factory import data_provider
from models.GraphMamba import Model
from scripts.diagnose_graphmamba_representations import command_args, locate_record

OUTPUT=ROOT/'logs'/'graphmamba_conditioned_memory_delta';DATASETS=('ETTm1','ETTm2');PRED=720;FULL=336;RECENT=96;POOL=16

def designs(x):
    recent=x[:,-RECENT:].permute(0,2,1)
    old=F.avg_pool1d(x[:,:-RECENT].permute(0,2,1),POOL,POOL)
    one=torch.ones(*recent.shape[:2],1,device=x.device)
    return torch.cat((recent,one),-1),torch.cat((old,recent,one),-1)

def fit_ridges(loader,n_vars):
    dims=(RECENT+1,15+RECENT+1);stats=[]
    for dim in dims:stats.append([torch.zeros((n_vars,dim,dim),device='cuda'),torch.zeros((n_vars,dim,PRED),device='cuda')])
    for batch_x,batch_y,*_ in loader:
        x=batch_x.float().cuda();y=batch_y[:,-PRED:,:].float().cuda().permute(0,2,1)
        for (xx,xy),design in zip(stats,designs(x)):
            xx+=torch.einsum('bvi,bvj->vij',design,design);xy+=torch.einsum('bvi,bvp->vip',design,y)
    weights=[]
    for xx,xy in stats:
        scale=xx[:,:-1,:-1].diagonal(dim1=-2,dim2=-1).mean();pen=torch.eye(xx.shape[-1],device='cuda')*1e-3*scale;pen[-1,-1]=0
        weights.append(torch.linalg.solve(xx+pen[None],xy))
    return weights

def baseline(model,batch_x,batch_y,bxm,bym):
    x=batch_x[:,-RECENT:].float().cuda();y=batch_y[:,-PRED:,:].float().cuda()
    decoder=torch.cat((batch_y[:,:48].float().cuda(),torch.zeros_like(y)),1)
    return y,model(x,bxm[:,-RECENT:].float().cuda(),decoder,bym.float().cuda())

def fit_alpha(model,loader,weights,n_vars):
    d2=torch.zeros(n_vars,dtype=torch.float64);dr=torch.zeros(n_vars,dtype=torch.float64)
    with torch.no_grad():
        for batch_x,batch_y,bxm,bym in loader:
            x=batch_x.float().cuda();recent,full=designs(x);delta=(torch.einsum('bvi,vip->bvp',full,weights[1])-torch.einsum('bvi,vip->bvp',recent,weights[0])).permute(0,2,1)
            y,pred=baseline(model,batch_x,batch_y,bxm,bym);residual=y-pred
            d2+=delta.double().cpu().square().sum((0,1));dr+=(delta*residual).double().cpu().sum((0,1))
    return dr/(d2+1e-12)

def evaluate(model,loader,weights,alpha):
    alpha=alpha.cuda().float();s={k:0. for k in ('b2','b1','c2','c1')};count=0
    with torch.no_grad():
        for batch_x,batch_y,bxm,bym in loader:
            x=batch_x.float().cuda();recent,full=designs(x);delta=(torch.einsum('bvi,vip->bvp',full,weights[1])-torch.einsum('bvi,vip->bvp',recent,weights[0])).permute(0,2,1)
            y,pred=baseline(model,batch_x,batch_y,bxm,bym);corrected=pred+alpha[None,None,:]*delta;be=y-pred;ce=y-corrected
            s['b2']+=be.square().sum().item();s['b1']+=be.abs().sum().item();s['c2']+=ce.square().sum().item();s['c1']+=ce.abs().sum().item();count+=y.numel()
    return {'base_mse':s['b2']/count,'base_mae':s['b1']/count,'corrected_mse':s['c2']/count,'corrected_mae':s['c1']/count,
            'mse_improvement_pct':100*(s['b2']-s['c2'])/s['b2'],'mae_improvement_pct':100*(s['b1']-s['c1'])/s['b1']}

def diagnose(dataset):
    record,checkpoint=locate_record(dataset,PRED);args=command_args(record['command']);loader_args=copy.copy(args);loader_args.seq_len=FULL;loader_args.batch_size=64
    torch.manual_seed(2021);model=Model(args).cuda().eval();model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True))
    _,train=data_provider(loader_args,'train');_,val=data_provider(loader_args,'val');weights=fit_ridges(train,args.enc_in);alpha=fit_alpha(model,train,weights,args.enc_in)
    return {'dataset':dataset,'pred_len':PRED,**evaluate(model,val,weights,alpha),'alpha':alpha.tolist(),'memory_tokens':15,
            'fit_split':'train','evaluation_split':'val','test_accessed':False,'checkpoint':str(checkpoint)}

def main():
    OUTPUT.mkdir(parents=True,exist_ok=True);rows=[]
    for dataset in DATASETS:
        print(f'Conditioned memory delta: {dataset}-{PRED}',flush=True);row=diagnose(dataset);rows.append(row);print(json.dumps(row,sort_keys=True),flush=True)
    with (OUTPUT/'summary.csv').open('w',newline='',encoding='utf-8') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return 0
if __name__=='__main__':raise SystemExit(main())
