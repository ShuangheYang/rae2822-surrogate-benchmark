# Publishing and GitHub Pages

The repository is designed for `ShuangheYang/rae2822-surrogate-benchmark`.

## Source repository

1. Create a **public**, empty GitHub repository named `rae2822-surrogate-benchmark`.
2. From this directory, run:

```bash
git init -b main
git add .
git commit -m "Release aerodynamic surrogate benchmark"
git remote add origin https://github.com/ShuangheYang/rae2822-surrogate-benchmark.git
git push -u origin main
```

If a Git repository already exists, skip initialization; if the remote already exists, inspect it before changing it. Authenticate using GitHub's browser/device flow or a credential manager. Do not place tokens in source files or commit credentials.

`runs/` and caches are ignored. Full datasets, checkpoints, and the research report are downloaded from pinned Hugging Face revisions instead of being committed to Git. The CFD-generation platform is outside the release directory.

## Free static site

In the repository, open **Settings → Pages**:

- Source: **Deploy from a branch**
- Branch: **main**
- Folder: **/docs**

After deployment succeeds, the site address is:

`https://shuangheyang.github.io/rae2822-surrogate-benchmark/`

This is the expected address after enabling Pages, not evidence that deployment has already completed. GitHub Pages serves static HTML/CSS/JavaScript. It supports the included airfoil explorer and can also host Reveal.js slides. It does not start Python, PyTorch or a CFD solver. No paid backend is configured.

To regenerate the static explorer data from the frozen study:

```bash
python run.py --run-dir runs/reference reference
python run.py --run-dir runs/reference explorer
```

Review and commit the updated `docs/assets/benchmark.json` together with any changed presentation source. The explorer displays saved predictions from the representative models. Generating new predictions from arbitrary user input would require a separately implemented browser inference engine or backend; it is not claimed by this static interface.
