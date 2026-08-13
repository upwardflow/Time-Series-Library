#!/usr/bin/env python3
"""Train-to-validation old-memory correction bound for frozen GraphMamba."""

from __future__ import annotations
import copy,csv,json,sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from data_provider.data_factory import data_provider
from models.GraphMamba import Model
from scripts.diagnose_graphmamba_representations import command_args, locate_record

OUTPUT=ROOT/'logs'/'graphmamba_old_memory_residual_bound';DATASETS=('ETTm1','ETTm2');PRED=720;FULL_HISTORY=336;RECENT=96;POOL=16;FEATURES=16

def memory_and_recent(x):
    recent=x[:,-RECENT:];means=recent.mean(1,keepdim=True);centered=recent-means
    stdev=torch.sqrt(torch.var(centered,dim=1,keepdim=True,unbiased=False)+1e-5)
    old=((x[:,:-RECENT]-means)/stdev).permute(0,2,1)
    pooled=F.avg_pool1d(old,kernel_size=POOL,stride=POOL)
    design=torch.cat((pooled,torch.ones(*pooled.shape[:2],1,device=x.device)),dim=-1)
    return design,recent,stdev

def fit(model,loader,n_vars):
    xx=torch.zeros((n_vars,FEATURES,FEATURES),dtype=torch.float64,device='cuda');xy=torch.zeros((n_vars,FEATURES,PRED),dtype=torch.float64,device='cuda')
    with torch.no_grad():
        for batch_x,batch_y,bxm,bym in loader:
            x=batch_x.float().cuda();y=batch_y[:,-PRED:,:].float().cuda();design,recent,stdev=memory_and_recent(x)
            decoder=torch.cat((batch_y[:,:48].float().cuda(),torch.zeros_like(y)),1)
            baseline=model(recent,bxm[:,-RECENT:].float().cuda(),decoder,bym.float().cuda())
            target=((y-baseline)/stdev).permute(0,2,1)
            xx+=torch.einsum('bvi,bvj->vij',design.double(),design.double());xy+=torch.einsum('bvi,bvp->vip',design.double(),target.double())
    scale=xx[:,:-1,:-1].diagonal(dim1=-2,dim2=-1).mean();pen=torch.eye(FEATURES,dtype=torch.float64,device='cuda')*1e-3*scale;pen[-1,-1]=0
    return torch.linalg.solve(xx+pen[None],xy)

def evaluate(model,loader,w):
    sums={k:0. for k in ('b2','b1','c2','c1')};count=0
    with torch.no_grad():
        for batch_x,batch_y,bxm,bym in loader:
            x=batch_x.float().cuda();y=batch_y[:,-PRED:,:].float().cuda();design,recent,stdev=memory_and_recent(x)
            decoder=torch.cat((batch_y[:,:48].float().cuda(),torch.zeros_like(y)),1)
            baseline=model(recent,bxm[:,-RECENT:].float().cuda(),decoder,bym.float().cuda())
            correction=torch.einsum('bvi,vip->bvp',design.double(),w).permute(0,2,1).float()*stdev
            be=y-baseline;ce=y-(baseline+correction)
            sums['b2']+=be.square().sum().item();sums['b1']+=be.abs().sum().item();sums['c2']+=ce.square().sum().item();sums['c1']+=ce.abs().sum().item();count+=y.numel()
    return {'base_mse':sums['b2']/count,'base_mae':sums['b1']/count,'corrected_mse':sums['c2']/count,'corrected_mae':sums['c1']/count,
            'mse_improvement_pct':100*(sums['b2']-sums['c2'])/sums['b2'],'mae_improvement_pct':100*(sums['b1']-sums['c1'])/sums['b1']}

def diagnose(dataset):
    record,checkpoint=locate_record(dataset,PRED);model_args=command_args(record['command']);loader_args=copy.copy(model_args);loader_args.seq_len=FULL_HISTORY;loader_args.batch_size=64
    torch.manual_seed(2021);model=Model(model_args).cuda().eval();model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True))
    _,train=data_provider(loader_args,'train');_,val=data_provider(loader_args,'val');w=fit(model,train,model_args.enc_in)
    return {'dataset':dataset,'pred_len':PRED,'recent_len':RECENT,'old_len':FULL_HISTORY-RECENT,'pool':POOL,'memory_tokens':15,
            **evaluate(model,val,w),'fit_split':'train','evaluation_split':'val','test_accessed':False,'checkpoint':str(checkpoint)}

def main():
    OUTPUT.mkdir(parents=True,exist_ok=True);rows=[]
    for dataset in DATASETS:
        print(f'Old-memory residual bound: {dataset}-{PRED}',flush=True);row=diagnose(dataset);rows.append(row);print(json.dumps(row,sort_keys=True),flush=True)
    with (OUTPUT/'summary.csv').open('w',newline='',encoding='utf-8') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return 0
if __name__=='__main__':raise SystemExit(main())
