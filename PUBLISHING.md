# PyPI Publishing Guide

This document outlines the complete PyPI publishing workflow for the `chain-of-thought-tool` package.

## 🔧 Setup Requirements

### 1. PyPI Trusted Publishing Setup

This project uses PyPI's Trusted Publishing (OIDC) for secure, token-free publishing.

#### Configure Trusted Publishers on PyPI:

1. **Create PyPI Account**: Register at https://pypi.org/ and https://test.pypi.org/
2. **Create Project**: Create the project `chain-of-thought-tool` on both PyPI instances
3. **Configure Trusted Publishing**:
   - Go to PyPI project settings → Publishing → Trusted Publishing
   - Add trusted publisher with:
     - **Owner**: `democratize-technology`
     - **Repository**: `chain-of-thought-tool`
     - **Workflow**: `publish.yml`
     - **Environment**: `pypi` (for production) and `testpypi` (for testing)

#### Configure GitHub Environments:

1. **Repository Settings** → Environments
2. **Create `pypi` environment**:
   - Protection rules: Require reviewer approval
   - Deployment branches: Selected branches → `main`
3. **Create `testpypi` environment**:
   - Protection rules: None (for easier testing)
   - Deployment branches: Any branch

## 🚀 Publishing Workflows

### Automatic Publishing (Recommended)

**Production Release**:
```bash
# Create and push a tag
git tag v0.1.1
git push origin v0.1.1

# GitHub will automatically:
# 1. Run tests
# 2. Build package
# 3. Create GitHub release
# 4. Publish to PyPI
```

**Test Publishing**:
```bash
# Manual workflow dispatch
gh workflow run publish.yml -f environment=testpypi
```

### Manual Publishing (Backup)

```bash
# Install build tools
pip install build twine

# Build package
python -m build

# Check package
twine check dist/*

# Upload to TestPyPI first
twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ chain-of-thought-tool

# Upload to PyPI
twine upload dist/*
```

## 📋 Release Checklist

### Pre-Release
- [ ] Update version in `chain_of_thought/__init__.py`
- [ ] Update CHANGELOG.md (if exists)
- [ ] Ensure all tests pass: `pytest`
- [ ] Ensure code quality: `black . && flake8`
- [ ] Test local build: `python -m build && twine check dist/*`
- [ ] Test installation: `pip install dist/*.whl`

### Release Process
- [ ] Create release tag: `git tag v0.x.y`
- [ ] Push tag: `git push origin v0.x.y`
- [ ] Monitor GitHub Actions workflow
- [ ] Verify PyPI upload
- [ ] Test installation from PyPI: `pip install chain-of-thought-tool`

### Post-Release
- [ ] Update documentation if needed
- [ ] Announce release (if appropriate)
- [ ] Monitor for issues

## 🔍 Troubleshooting

### Common Issues

**Trusted Publishing Not Working**:
1. Verify OIDC configuration on PyPI matches exactly
2. Check GitHub environment settings
3. Ensure workflow has `id-token: write` permission

**Build Failures**:
1. Check Python version compatibility (3.8-3.12)
2. Verify all imports work: `python -c "import chain_of_thought"`
3. Check package metadata: `twine check dist/*`

**Version Conflicts**:
1. PyPI doesn't allow overwriting versions
2. Increment version number and create new release
3. Use dev versions for testing: `0.1.0.dev1`

### Testing Workflow

```bash
# Test the workflow without publishing
gh workflow run test.yml

# Test publishing to TestPyPI
gh workflow run publish.yml -f environment=testpypi

# Verify TestPyPI installation
pip install --index-url https://test.pypi.org/simple/ chain-of-thought-tool
```

## 🛡️ Security Best Practices

1. **No Manual Tokens**: Use Trusted Publishing exclusively
2. **Environment Protection**: Require approval for production releases
3. **Branch Protection**: Only release from protected branches
4. **Version Pinning**: Pin all GitHub Actions to specific versions
5. **Artifact Verification**: Always run `twine check` before upload

## 📊 Package Metrics

Monitor package health:
- **Downloads**: https://pypistats.org/packages/chain-of-thought-tool
- **Security**: https://pypi.org/project/chain-of-thought-tool/
- **Dependencies**: No external dependencies (zero-dependency policy)

## 🔄 Automation Features

### Continuous Integration
- **Multi-Python Testing**: Tests on Python 3.8-3.12
- **Code Quality**: Black formatting, Flake8 linting
- **Package Validation**: Build and installation tests

### Continuous Deployment
- **Trusted Publishing**: Secure, token-free deployment
- **Environment Gates**: Manual approval for production
- **Artifact Handling**: Build once, deploy everywhere
- **Release Automation**: Auto-generated release notes

### Monitoring
- **Build Status**: GitHub Actions badges
- **Test Coverage**: Coverage reports (when test suite is ready)
- **Package Quality**: Automated quality checks

## 📞 Support

For publishing issues:
1. Check GitHub Actions logs
2. Review PyPI project settings
3. Verify trusted publisher configuration
4. Test with TestPyPI first

For package issues:
1. Create GitHub issue
2. Include version and environment details
3. Provide minimal reproduction case