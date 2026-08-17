#!/usr/bin/env python3
"""Frozen split-validation upper bound for seasonal/trend forecast fusion."""

from __future__ import annotations

import csv
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_provider.data_factory import data_provider
from models.GraphMamba import Model
from scripts.diagnose_graphmamba_representations import command_args, locate_record

OUTPUT = ROOT / "logs" / "graphmamba_seasonal_trend_fusion_bound"
DATASETS = ("ETTm1", "ETTm2")


def new_stats(n_vars: int) -> dict[str, torch.Tensor]:
    return {
        "xtx": torch.zeros((n_vars, 2, 2), dtype=torch.float64),
        "xtr": torch.zeros((n_vars, 2), dtype=torch.float64),
        "rtr": torch.zeros(n_vars, dtype=torch.float64),
        "count": torch.zeros(n_vars, dtype=torch.float64),
        "seasonal2": torch.zeros(n_vars, dtype=torch.float64),
        "trend2": torch.zeros(n_vars, dtype=torch.float64),
        "seasonal_trend": torch.zeros(n_vars, dtype=torch.float64),
    }


def update(store, seasonal, trend, residual) -> None:
    x = torch.stack((seasonal, trend), dim=-1).detach().double().cpu()
    residual = residual.detach().double().cpu()
    store["xtx"] += torch.einsum("btvi,btvj->vij", x, x)
    store["xtr"] += torch.einsum("btvi,btv->vi", x, residual)
    store["rtr"] += residual.square().sum(dim=(0, 1))
    store["count"] += residual.shape[0] * residual.shape[1]
    store["seasonal2"] += seasonal.detach().double().cpu().square().sum(dim=(0, 1))
    store["trend2"] += trend.detach().double().cpu().square().sum(dim=(0, 1))
    store["seasonal_trend"] += (seasonal * trend).detach().double().cpu().sum(dim=(0, 1))


def diagnose(dataset: str, pred_len: int) -> tuple[dict, list[dict]]:
    record, checkpoint = locate_record(dataset, pred_len)
    args = command_args(record["command"])
    torch.manual_seed(2021)
    model = Model(args).cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
    _, loader = data_provider(args, "val")
    split = len(loader) // 2
    calibration, evaluation = new_stats(args.enc_in), new_stats(args.enc_in)
    equivalence_max_abs = 0.0

    with torch.no_grad():
        for batch_index, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
            x = batch_x.float().cuda(); y = batch_y[:, -pred_len:, :].float().cuda()
            decoder = torch.cat((batch_y[:, :args.label_len].float().cuda(), torch.zeros_like(y)), dim=1)
            baseline = model(x, batch_x_mark.float().cuda(), decoder, batch_y_mark.float().cuda())
            means = x.mean(dim=1, keepdim=True); centered = x-means
            stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False)+1e-5)
            seasonal_input, trend_input = model.decomposition(centered/stdev)
            trend_normalized = model.trend_projection(trend_input.permute(0,2,1)).permute(0,2,1)
            seasonal_input = seasonal_input.permute(0,2,1)
            tokens = torch.cat((
                model.long_patch_embedding(seasonal_input)+model.variable_embedding,
                model.short_patch_embedding(seasonal_input)+model.variable_embedding,
            ),dim=-1)
            seasonal_normalized = model.head(model.encoder(tokens)+model.graph_mixer(tokens))
            reconstructed = (seasonal_normalized+trend_normalized)*stdev+means
            equivalence_max_abs=max(equivalence_max_abs,(reconstructed-baseline).abs().max().item())
            seasonal_contribution=seasonal_normalized*stdev
            trend_contribution=trend_normalized*stdev
            update(calibration if batch_index<split else evaluation,
                   seasonal_contribution,trend_contribution,y-baseline)

    eye=torch.eye(2,dtype=torch.float64).unsqueeze(0)
    coefficients=torch.linalg.solve(calibration["xtx"]+1e-8*eye,calibration["xtr"].unsqueeze(-1)).squeeze(-1)
    corrected_sse=evaluation["rtr"].clone()
    for var in range(args.enc_in):
        beta=coefficients[var]
        corrected_sse[var]+=beta@evaluation["xtx"][var]@beta-2*beta@evaluation["xtr"][var]
    base_mse=evaluation["rtr"].sum().item()/evaluation["count"].sum().item()
    corrected_mse=corrected_sse.sum().item()/evaluation["count"].sum().item()
    cosine=evaluation["seasonal_trend"]/torch.sqrt(evaluation["seasonal2"]*evaluation["trend2"]+1e-12)
    names=["HUFL","HULL","MUFL","MULL","LUFL","LULL","OT"]
    variables=[]
    for var,name in enumerate(names):
        base_var=evaluation["rtr"][var]/evaluation["count"][var]
        corrected_var=corrected_sse[var]/evaluation["count"][var]
        variables.append({"dataset":dataset,"variable":name,"seasonal_delta":coefficients[var,0].item(),
            "trend_delta":coefficients[var,1].item(),"seasonal_multiplier":1+coefficients[var,0].item(),
            "trend_multiplier":1+coefficients[var,1].item(),"seasonal_trend_cosine":cosine[var].item(),
            "base_mse":base_var.item(),"corrected_mse":corrected_var.item(),
            "improvement_pct":100*(base_var-corrected_var).item()/base_var.item()})
    summary={"dataset":dataset,"pred_len":pred_len,"base_mse":base_mse,"corrected_mse":corrected_mse,
        "improvement_pct":100*(base_mse-corrected_mse)/base_mse,"equivalence_max_abs":equivalence_max_abs,
        "calibration_batches":split,"evaluation_batches":len(loader)-split,"test_accessed":False,
        "mean_seasonal_trend_cosine":cosine.mean().item()}
    return summary,variables


def main()->int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--pred-len",type=int,default=720,choices=(96,192,336,720))
    args=parser.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True); summaries=[]; variables=[]
    for dataset in DATASETS:
        print(f"Seasonal/trend diagnosis: {dataset}-{args.pred_len}",flush=True)
        summary,rows=diagnose(dataset,args.pred_len); summaries.append(summary); variables.extend(rows)
        (OUTPUT/f"{dataset.lower()}_{args.pred_len}.json").write_text(json.dumps({"summary":summary,"variables":rows},indent=2)+"\n")
        print(json.dumps(summary,sort_keys=True),flush=True)
    for filename,rows in ((f"summary_{args.pred_len}.csv",summaries),(f"variables_{args.pred_len}.csv",variables)):
        with (OUTPUT/filename).open("w",newline="",encoding="utf-8") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
