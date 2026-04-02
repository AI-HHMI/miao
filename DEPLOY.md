# Publishing miao to PyPI

## Prerequisites

```bash
pip install build twine
```

You also need a PyPI account. Create one at https://pypi.org/account/register/ and generate an API token at https://pypi.org/manage/account/token/.

## 1. Update the version

Edit `pyproject.toml` and bump the version:

```toml
[project]
version = "0.1.1"
```

## 2. Build

```bash
rm -rf dist/
python -m build
```

This creates `dist/miao-<version>.tar.gz` (sdist) and `dist/miao-<version>-py3-none-any.whl` (wheel).

## 3. (Optional) Test on TestPyPI

```bash
twine upload --repository testpypi dist/*
```

Then verify:

```bash
pip install --index-url https://test.pypi.org/simple/ miao
```

## 4. Upload to PyPI

```bash
twine upload dist/*
```

You will be prompted for credentials. Use `__token__` as the username and your API token as the password.

To avoid entering credentials each time, create `~/.pypirc`:

```ini
[pypi]
username = __token__
password = pypi-<your-token>
```
