# GraphMamba–CMRHM BibTeX 核验报告

## 结果

- 合并后初稿/激进修订版唯一 citation keys：17
- BibTeX 条目：17
- 已核验：17
- 缺失键：0
- 未使用条目：0
- 重复 DOI：0
- 重复题名：0
- Pandoc `citeproc` 解析：通过
- Biber 数据模型校验：通过（0 errors）

输出文件：`GraphMamba_CMRHM_references.bib`

## 逐条核验

| Citation key | 状态 | 正式版本 | 标识符/官方入口 | 核验说明 |
|---|---|---|---|---|
| `Chen2026Wrinkles` | verified | Neurocomputing 676 (2026), 133021 | DOI `10.1016/j.neucom.2026.133021` | Crossref 与本地正式 PDF 一致 |
| `Chen2026GMamba` | verified | Neurocomputing 680 (2026), 133280 | DOI `10.1016/j.neucom.2026.133280` | Crossref 与出版社页面一致 |
| `Duan2026MPGTimer` | verified | Neurocomputing 700 (2026), 134375 | DOI `10.1016/j.neucom.2026.134375` | Crossref 与出版社页面一致 |
| `Gu2024Mamba` | verified | COLM 2024 | OpenReview `tEYskw1VY2`; arXiv `2312.00752` | 使用正式会议年份 2024，不采用预印本年份 2023 |
| `Hu2024Attraos` | verified | NeurIPS 2024, vol. 37, pp. 20786--20818 | DOI `10.52202/079017-0655` | Crossref 与 NeurIPS 官方页面一致 |
| `Huang2023CrossGNN` | verified | NeurIPS 2023, vol. 36, pp. 46885--46902 | DOI `10.52202/075280-2031` | Crossref 与 NeurIPS 官方页面一致 |
| `Liu2024iTransformer` | verified | ICLR 2024 | OpenReview `JePfAI8fah` | DBLP 与 OpenReview 会议版本一致 |
| `Liu2022TPGNN` | verified | NeurIPS 2022, vol. 35, pp. 19414--19426 | DOI `10.52202/068431-1411` | Crossref 与 NeurIPS 官方页面一致 |
| `Mo2026DiM` | verified | Neurocomputing 659 (2026), 131777 | DOI `10.1016/j.neucom.2025.131777` | Crossref 与本地正式 PDF 一致；DOI 年份 2025、卷年份 2026 均正确 |
| `Nie2023PatchTST` | verified | ICLR 2023 | OpenReview `Jbdc0vTOcol` | 使用正式会议版本，不以 arXiv 记录替代 |
| `Wang2024TimeMixer` | verified | ICLR 2024 | OpenReview `7oLshfEIC2` | DBLP、OpenReview 与作者官方仓库一致 |
| `Wang2025SMamba` | verified | Neurocomputing 619 (2025), 129178 | DOI `10.1016/j.neucom.2024.129178` | Crossref 与出版社页面一致 |
| `Wu2021Autoformer` | verified | NeurIPS 2021, vol. 34, pp. 22419--22430 | NeurIPS 官方页面 | 正式论文集未提供常规论文 DOI |
| `Wu2023TimesNet` | verified | ICLR 2023 | OpenReview `ju_Uqw384Oq` | DBLP 与 OpenReview 会议版本一致 |
| `Zhang2025MEW` | verified | NeurIPS 2025, vol. 38, pp. 93680--93704 | DOI `10.52202/085713-2817` | Crossref 与 NeurIPS 官方页面一致 |
| `Zhou2021Informer` | verified | AAAI 2021, vol. 35(12), pp. 11106--11115 | DOI `10.1609/aaai.v35i12.17325` | Crossref 与 AAAI 官方页面一致 |
| `Zhou2026SSMGNN` | verified | Neurocomputing 666 (2026), 132295 | DOI `10.1016/j.neucom.2025.132295` | Crossref 与出版社页面一致 |

## 版本选择原则

1. 已正式发表的 ICLR、COLM、NeurIPS 论文使用会议版本，而不是 arXiv 预印本版本。
2. 没有正式 DOI 的会议论文保留 OpenReview 或 NeurIPS 官方 URL，不编造 DOI。
3. Neurocomputing 的文章号放在 `pages` 字段，兼容传统 BibTeX 和常见 Elsevier 模板。
4. citation keys 完全沿用正文，正文无需因为本次补库而改名。

## 使用方式

Pandoc Markdown 可在 YAML front matter 中加入：

```yaml
bibliography: GraphMamba_CMRHM_references.bib
```

如果从项目根目录编译，则使用：

```yaml
bibliography: paper/GraphMamba_CMRHM_references.bib
```

LaTeX/BibTeX 可使用：

```latex
\bibliography{paper/GraphMamba_CMRHM_references}
```

注：Pandoc 端到端测试中出现了正文 LaTeX 数学公式向纯文本转换的提示，但没有文献解析警告；这些提示与 BibTeX 数据无关。

## 后续边界

本文件覆盖合并后初稿与激进修订版实际出现的 17 个 citation keys；相较合并前原稿新增 MPGTimer、G-Mamba、SSMGNN、S-Mamba、TPGNN 和 CrossGNN 六项正式文献。主 BibTeX 未加入正文未使用的候选条目。
