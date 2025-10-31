# Contract Test Template (drop-in)

Purpose: keep domain packs and Sentinel aligned with Summit.OS canonical contracts.

Run locally:
- pip install -r ci/templates/requirements.txt
- pytest ci/templates/contract_test_template.py -q

Optional live API validation:
- export API_BASE_URL=http://localhost:8000
- pytest ci/templates/contract_test_template.py -q

CI integration (GitHub Actions example): see ci/templates/gh-actions/contract-tests.yml. Copy into .github/workflows/ to activate.
