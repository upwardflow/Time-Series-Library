# TimeRole 论文图片

本目录集中管理论文正文使用的图片及其可编辑源文件。

总体架构图采用 Draw.io 原生工作流，统一使用以下文件名：

- `Fig1_TimeRole_Architecture.drawio`：draw.io 可编辑源文件。
- `Fig1_TimeRole_Architecture.pdf`：投稿与正式排版使用的矢量文件。
- `Fig1_TimeRole_Architecture.svg`：矢量预览及后续编辑文件。
- `Fig1_TimeRole_Architecture.png`：快速审阅用预览图，不作为首选投稿文件。
- `Fig1_TimeRole_Architecture.html`：浏览器查看页面。

后续图片沿用 `FigN_TimeRole_内容名称` 的命名方式。图片标题和详细图注保留在论文正文中，不写入图像画布。

`Fig1_TimeRole_Architecture.drawio` 是唯一主源文件；PDF、SVG 和 PNG 均由 diagrams.net Desktop 直接导出，HTML 仅作为浏览器查看入口。

项目内 Draw.io Desktop 启动入口为 `tools/drawio/drawio`。例如：

```bash
tools/drawio/drawio paper/picture/Fig1_TimeRole_Architecture.drawio
```
