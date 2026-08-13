#!/usr/bin/env python3
"""Fair-target ridge probe for long-history predictability at horizon 720."""

from __future__ import annotations

import csv, json
from pathlib import Path
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'dataset'/'ETT-small';OUTPUT=ROOT/'logs'/'graphmamba_history_length_predictability'
DATASETS=('ETTm1','ETTm2');LENGTHS=(96,192,336);PRED_LEN=720;TRAIN_END=12*30*24*4;VAL_END=TRAIN_END+4*30*24*4

def load(dataset):
    values=torch.tensor(pd.read_csv(DATA/f'{dataset}.csv').iloc[:,1:].to_numpy(),dtype=torch.float64)
    mean=values[:TRAIN_END].mean(0);std=values[:TRAIN_END].std(0,unbiased=False).clamp_min(1e-8)
    return (values-mean)/std

def batches(starts,batch=256):
    for i in range(0,len(starts),batch):yield starts[i:i+batch]

def design_targets(values,starts,length):
    # starts are target start indices; history ends immediately before each start.
    offsets=torch.arange(-length,0)
    future=torch.arange(PRED_LEN)
    x=values[starts[:,None]+offsets[None,:]].permute(0,2,1)
    y=values[starts[:,None]+future[None,:]].permute(0,2,1)
    ones=torch.ones((*x.shape[:2],1),dtype=x.dtype)
    return torch.cat((x,ones),dim=-1),y

def fit(values,length):
    n_vars=values.shape[1];dim=length+1
    xtx=torch.zeros((n_vars,dim,dim),dtype=torch.float64,device='cuda')
    xty=torch.zeros((n_vars,dim,PRED_LEN),dtype=torch.float64,device='cuda')
    starts=torch.arange(max(LENGTHS),TRAIN_END-PRED_LEN+1)
    for indices in batches(starts):
        x,y=design_targets(values,indices,length);x=x.cuda();y=y.cuda()
        xtx+=torch.einsum('bvi,bvj->vij',x,x);xty+=torch.einsum('bvi,bvp->vip',x,y)
    scale=xtx[:,:-1,:-1].diagonal(dim1=-2,dim2=-1).mean()
    penalty=torch.eye(dim,dtype=torch.float64,device='cuda')*(1e-3*scale);penalty[-1,-1]=0
    return torch.linalg.solve(xtx+penalty[None],xty),float(1e-3*scale)

def evaluate(values,length,weights):
    starts=torch.arange(TRAIN_END,VAL_END-PRED_LEN+1);sse=torch.zeros(values.shape[1],dtype=torch.float64);count=0
    for indices in batches(starts):
        x,y=design_targets(values,indices,length);prediction=torch.einsum('bvi,vip->bvp',x.cuda(),weights)
        sse+=(prediction-y.cuda()).square().sum((0,2)).cpu();count+=y.shape[0]*PRED_LEN
    per=sse/count;return per.mean().item(),per.tolist()

def main():
    OUTPUT.mkdir(parents=True,exist_ok=True);summaries=[];variables=[];names=('HUFL','HULL','MUFL','MULL','LUFL','LULL','OT')
    for dataset in DATASETS:
        values=load(dataset);rows=[]
        for length in LENGTHS:
            weights,ridge=fit(values,length);mse,per=evaluate(values,length,weights)
            row={'dataset':dataset,'seq_len':length,'pred_len':PRED_LEN,'validation_mse':mse,'ridge':ridge,
                 'train_samples':TRAIN_END-PRED_LEN-max(LENGTHS)+1,'validation_samples':VAL_END-PRED_LEN-TRAIN_END+1}
            rows.append(row)
            for name,value in zip(names,per):variables.append({**row,'variable':name,'variable_mse':value})
            print(json.dumps(row,sort_keys=True),flush=True)
        current=rows[0]['validation_mse']
        for row in rows:row['vs_96_improvement_pct']=100*(current-row['validation_mse'])/current
        summaries.extend(rows)
    for filename,rows in (('summary.csv',summaries),('variables.csv',variables)):
        with (OUTPUT/filename).open('w',newline='',encoding='utf-8') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return 0
if __name__=='__main__':raise SystemExit(main())
