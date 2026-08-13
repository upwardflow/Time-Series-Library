#!/usr/bin/env python3
"""Probe dense-recent plus pooled-older history representations."""

from __future__ import annotations
import csv,json
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'dataset'/'ETT-small';OUTPUT=ROOT/'logs'/'graphmamba_compressed_history_memory'
DATASETS=('ETTm1','ETTm2');PRED=720;TRAIN_END=34560;VAL_END=46080;MAX_HISTORY=336
VARIANTS=(('recent96',None),('old_pool16',16),('old_pool8',8),('old_pool4',4),('raw336',1))

def load(name):
    x=torch.tensor(pd.read_csv(DATA/f'{name}.csv').iloc[:,1:].to_numpy(),dtype=torch.float64);m=x[:TRAIN_END].mean(0);s=x[:TRAIN_END].std(0,unbiased=False)
    return (x-m)/s.clamp_min(1e-8)

def feature(values,starts,pool):
    recent=values[starts[:,None]+torch.arange(-96,0)[None]].permute(0,2,1)
    if pool is None:return recent
    old=values[starts[:,None]+torch.arange(-336,-96)[None]].permute(0,2,1)
    if pool>1:old=F.avg_pool1d(old,kernel_size=pool,stride=pool)
    return torch.cat((old,recent),-1)

def target(values,starts):return values[starts[:,None]+torch.arange(PRED)[None]].permute(0,2,1)
def chunks(starts,n=256):
    for i in range(0,len(starts),n):yield starts[i:i+n]

def fit(values,pool):
    dim=96+(0 if pool is None else 240//pool)+1;n=values.shape[1]
    xx=torch.zeros((n,dim,dim),device='cuda',dtype=torch.float64);xy=torch.zeros((n,dim,PRED),device='cuda',dtype=torch.float64)
    for starts in chunks(torch.arange(MAX_HISTORY,TRAIN_END-PRED+1)):
        x=feature(values,starts,pool);x=torch.cat((x,torch.ones(*x.shape[:2],1,dtype=x.dtype)),-1).cuda();y=target(values,starts).cuda()
        xx+=torch.einsum('bvi,bvj->vij',x,x);xy+=torch.einsum('bvi,bvp->vip',x,y)
    scale=xx[:,:-1,:-1].diagonal(dim1=-2,dim2=-1).mean();pen=torch.eye(dim,device='cuda',dtype=torch.float64)*1e-3*scale;pen[-1,-1]=0
    return torch.linalg.solve(xx+pen[None],xy),dim

def evaluate(values,pool,w):
    se=torch.zeros(values.shape[1],dtype=torch.float64);count=0
    for starts in chunks(torch.arange(TRAIN_END,VAL_END-PRED+1)):
        x=feature(values,starts,pool);x=torch.cat((x,torch.ones(*x.shape[:2],1,dtype=x.dtype)),-1).cuda();y=target(values,starts).cuda()
        se+=(torch.einsum('bvi,vip->bvp',x,w)-y).square().sum((0,2)).cpu();count+=y.shape[0]*PRED
    per=se/count;return per.mean().item(),per.tolist()

def main():
    OUTPUT.mkdir(parents=True,exist_ok=True);rows=[];variables=[];names=('HUFL','HULL','MUFL','MULL','LUFL','LULL','OT')
    for dataset in DATASETS:
        values=load(dataset);local=[]
        for name,pool in VARIANTS:
            w,dim=fit(values,pool);mse,per=evaluate(values,pool,w);row={'dataset':dataset,'variant':name,'input_features':dim-1,'validation_mse':mse};local.append(row)
            for variable,value in zip(names,per):variables.append({**row,'variable':variable,'variable_mse':value})
            print(json.dumps(row,sort_keys=True),flush=True)
        base=local[0]['validation_mse'];full_gain=base-local[-1]['validation_mse']
        for row in local:
            gain=base-row['validation_mse'];row['vs_recent96_improvement_pct']=100*gain/base;row['fraction_of_raw336_gain']=gain/full_gain if full_gain>0 else 0
        rows.extend(local)
    for fn,data in (('summary.csv',rows),('variables.csv',variables)):
        with (OUTPUT/fn).open('w',newline='',encoding='utf-8') as h:w=csv.DictWriter(h,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    return 0
if __name__=='__main__':raise SystemExit(main())
