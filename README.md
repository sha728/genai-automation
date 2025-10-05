# Open-Source LLM Text Analysis (Free)

Analyzes customer feedback using free, open-source LLMs from Hugging Face.

### Features
- Sentiment classification (`distilbert-base-uncased-sst-2-english`)
- Summarization (`facebook/bart-large-cnn`)
- Simple keyword-based issue detection

### Run
```bash
pip install transformers torch pandas
jupyter notebook text_analysis_free.ipynb
