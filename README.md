# E-commerce Customer Support Agent System

ML recommendations, RAG search, LangChain agents.

Screenshots:

https://github.com/Leonid328/telco-churn-mlops-synthetic-06/tree/develop/docs/screenshots

```bash
# one-command setup
./scripts/setup/init_project.sh --quick
```

```bash
# clone and run locally
git clone https://github.com/Leonid328/ecommerce-support-cicd.git
cd ecommerce-support-cicd
python scripts/generate_data.py
pip install -r requirements.txt
streamlit run src/app.py
# or
docker compose up --build
```

http://localhost:8501

- ML Service: http://localhost:8001
- RAG Service: http://localhost:8002
- Agent Service: http://localhost:8003
- MLflow: http://localhost:5000
- Grafana: http://localhost:3000

[Architecture](docs/architecture.md)

[API Reference](docs/api-reference.md)

Apache 2.0
