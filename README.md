# MailMind — AI Email Classifier & Reply Drafter

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red?logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![CI](https://github.com/YOUR_USERNAME/mailmind/actions/workflows/ci.yml/badge.svg)

Paste any business email and get instant structured intelligence: category, priority, sentiment, and a professional reply — drafted automatically and tailored to the tone of the original message.

**Works with OpenAI, Anthropic Claude, Google Gemini, or a fully local Ollama model.**

---

## Features

- **Multi-provider LLM support** — OpenAI, Anthropic, Google Gemini, or local Ollama
- **Structured classification** — category, priority (with SLA), and sentiment via Pydantic schema
- **Auto-toned drafts** — reply tone is inferred from category; overridable manually
- **AI reasoning** — explains *why* it classified the email the way it did
- **Streaming draft generation** — response types out token by token
- **4 built-in example emails** — one click to populate and test
- **Clean dark UI** — polished Streamlit interface

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                       MailMind                           │
│                                                          │
│  ┌─────────────┐    ┌────────────────────────────────┐   │
│  │  core/      │    │  app.py (Streamlit UI)          │   │
│  │             │    │                                 │   │
│  │ schemas.py  │    │  Sidebar: provider selector     │   │
│  │  EmailAnalysis   │          example emails         │   │
│  │  RiskLevel  │    │                                 │   │
│  │  Category   │    │  Main:   email text input       │   │
│  │             │    │          classification grid    │   │
│  │ classifier  │───▶│          key points + reasoning │   │
│  │  chain      │    │          draft reply editor     │   │
│  │             │    └────────────────────────────────┘   │
│  │ drafter     │                                          │
│  │  chain      │    ┌────────────────────────────────┐   │
│  │  TONE_MAP   │    │  llm_config.py                  │   │
│  └─────────────┘    │  ├─ render_llm_selector()       │   │
│                     │  └─ build_llm()                 │   │
│                     └────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

**Classification pipeline:**
```
Email text → LLM (structured output) → Pydantic validation → UI cards
                                                 │
                                         Category + tone
                                                 │
                                         Drafter chain → Streamed reply
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/mailmind.git
cd mailmind
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

### 3. Configure a provider

| Provider | Key format | Where to get it |
|---|---|---|
| OpenAI | `sk-...` | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| Anthropic | `sk-ant-...` | [console.anthropic.com/keys](https://console.anthropic.com/keys) |
| Google Gemini | `AIza...` | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| **Local Ollama** | None | Run `ollama serve` locally |

**Using Ollama (free, private):**
```bash
ollama pull llama3.2
ollama serve
```
Select **"Local (Ollama)"** in the sidebar. No API costs.

---

## Email categories

| Category | Icon | Auto-detected tone |
|---|---|---|
| Support | 🛠️ | Helpful and clear |
| Sales | 💼 | Professional and value-focused |
| Complaint | ⚠️ | Empathetic and solution-focused |
| Inquiry | ❓ | Informative and engaging |
| Internal | 🏢 | Direct and collegial |
| Spam | 🚫 | Brief and non-committal |

---

## Docker

```bash
docker build -t mailmind .
docker run -p 8501:8501 mailmind
```

---

## Deploy to Streamlit Community Cloud (free)

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your fork → branch `main` → main file `app.py`
4. Deploy — no secrets needed, users enter their own key

---

## Development

```bash
pip install -e ".[dev]"
make test        # run tests
make test-cov    # with coverage report
make lint        # ruff checks
```

### Project structure

```
mailmind/
├── core/
│   ├── __init__.py       # Public API
│   ├── schemas.py        # Pydantic models (EmailAnalysis, enums)
│   ├── classifier.py     # Classification LCEL chain
│   └── drafter.py        # Reply drafting chain + TONE_MAP
├── tests/
│   ├── conftest.py       # Shared fixtures
│   ├── test_schemas.py   # Pydantic schema validation tests
│   └── test_chains.py    # Chain builder tests (mocked LLM)
├── .github/workflows/
│   └── ci.yml            # GitHub Actions: test + lint
├── app.py                # Streamlit UI entry point
├── llm_config.py         # Multi-provider LLM selector widget
├── pyproject.toml        # Project metadata + tool config
├── Makefile              # Developer shortcuts
├── Dockerfile            # Container image
├── .env.example          # API key template
└── requirements.txt      # Pinned dependencies
```

---

## Contributing

1. Fork and create a feature branch
2. Run `make test` and `make lint` before opening a PR
3. One feature or fix per PR

---

## License

MIT — see [LICENSE](LICENSE).

---

*Built with [LangChain](https://langchain.com), [Streamlit](https://streamlit.io), and [Pydantic](https://docs.pydantic.dev).*
