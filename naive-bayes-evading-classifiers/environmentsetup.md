# Environment Setup

A dedicated project directory and Python virtual environment were created to isolate dependencies.

```bash
mkdir goodwords_attack
cd goodwords_attack
python3 -m venv venv
source venv/bin/activate
```

Required dependencies were installed:

```bash
pip3 install scikit-learn numpy requests
```

The target service URL was exported as an environment variable:

```bash
export BASE_URL="http://94.237.120.112:51238"
```

Service availability was verified:

```bash
curl -s "$BASE_URL/health" | jq
```

```json
{
  "service": "skills_assessment_lab",
  "status": "healthy"
}
```
