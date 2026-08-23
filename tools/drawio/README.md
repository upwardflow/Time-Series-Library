# Draw.io Desktop

- Version: 31.1.8
- Platform: Linux x86_64
- Source: official `jgraph/drawio-desktop` GitHub release
- AppImage SHA-256: `19b208eb2b54fd6dda64fbfe403379458f12e1e0265b66c3386d1c021085efa2`

The verified AppImage was extracted locally because this environment does not provide FUSE 2. Start Draw.io with:

```bash
tools/drawio/drawio [diagram.drawio]
```

Example export:

```bash
tools/drawio/drawio --export --format svg --output output.svg input.drawio
```
