# Publishing and GitHub Pages

Repository: [ShuangheYang/rae2822-surrogate-benchmark](https://github.com/ShuangheYang/rae2822-surrogate-benchmark).

Website: [Live prediction and explorer](https://shuangheyang.github.io/rae2822-surrogate-benchmark/#predict).

## Source repository

For the existing repository, commit reviewed changes and push them to `main`:

```bash
git add .
git commit -m "Describe the change"
git push origin main
```

Check `git status` and `git diff --cached` before committing. Authentication uses GitHub's device flow or a credential manager; credentials do not belong in the repository.

`runs/` and caches are ignored. Full datasets, Python checkpoints and the research report are downloaded from pinned Hugging Face revisions. The small browser weight exports are committed under `docs/assets/inference/`. The CFD-generation platform is outside this repository.

## Free static site

In the repository, open **Settings → Pages**:

- Source: **Deploy from a branch**
- Branch: **main**
- Folder: **/docs**

GitHub Pages serves HTML, CSS, JavaScript and model assets. The visitor's browser fits coordinates and evaluates the trained models in a Web Worker. There is no Python service, GPU rental, API key or paid inference endpoint. Input coordinates and coefficients are not uploaded. Repository updates are available after the Pages deployment finishes in the Actions tab.

To regenerate the static explorer data from the frozen study:

```bash
python run.py --run-dir runs/reference reference
python run.py --run-dir runs/reference explorer
```

The explorer reads `docs/assets/benchmark.json`, which contains saved predictions. The live prediction panel instead evaluates the model weights for the submitted geometry.

## Browser model assets

To reproduce the checked-in model exports:

```bash
python run.py --run-dir runs/reference models
python run.py --run-dir runs/reference browser-models
node tests/browser_core.mjs
```

The exporter verifies the checkpoint hashes against `configs/assets.lock.json`. It writes little-endian float64 arrays and a JSON manifest containing the original checkpoint URLs, hashes, scalers and network/kernel definitions. No fitting, distillation or quantization takes place. Content-addressed binary filenames prevent an old cached model from being confused with a new manifest.

Commit the manifest and both binary files together with any numerical implementation changes. Recheck prediction parity and coordinate handling before publishing. Browser tests run in GitHub Actions without Python, training, tokens or external model downloads.

## Local preview

```bash
python -m http.server 8000 --directory docs
```

Open `http://localhost:8000/#predict`. A current browser and HTTP on localhost or HTTPS online are required for modules, workers and SHA-256 verification. Opening the HTML directly as a local file is unsupported.
