from transformers import pipeline
import pandas as pd

# --- Sample feedback data ---
data = pd.DataFrame({
    "customer_feedback": [
        "The food was cold and took too long to arrive.",
        "Amazing service and very fast delivery!",
        "The app crashed while ordering.",
        "Delivery person was polite but forgot my drink.",
        "Loved the spicy chicken, it was perfect!",
        "The restaurant sent the wrong dish.",
        "App is very slow and often freezes.",
        "Delivery was quick, and the driver was courteous.",
        "Food packaging was messy, some items spilled.",
        "Great variety of options, very satisfied!",
        "Delivery arrived 30 minutes late without any notification.",
        "The app suggested wrong dishes, confusing interface.",
        "Driver called to confirm address, appreciated that!",
        "Food portion was too small for the price.",
        "Excellent discounts and offers this week.",
        "Order was canceled without reason, very frustrating.",
        "Fresh and tasty food, loved the presentation.",
        "The payment failed multiple times on the app.",
        "Driver was friendly but took a longer route.",
        "App notifications are inconsistent, sometimes delayed."
    ]
})

# --- Load open-source models ---
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def classify_issue(text):
    text_lower = text.lower()
    if "app" in text_lower or "crash" in text_lower:
        return "App"
    elif "delivery" in text_lower:
        return "Delivery"
    elif "food" in text_lower:
        return "Food"
    else:
        return "Other"

def analyze_feedback(text):
    sentiment = sentiment_analyzer(text)[0]['label']
    summary = summarizer(text, max_length=20, min_length=5, do_sample=False)[0]['summary_text']
    issue_type = classify_issue(text)
    return sentiment, issue_type, summary

data[['Sentiment', 'Issue Type', 'Summary']] = data['customer_feedback'].apply(
    lambda x: pd.Series(analyze_feedback(x))
)

data.to_csv("feedback_analysis.csv", index=False)
print(data)
