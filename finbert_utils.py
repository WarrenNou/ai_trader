from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Tuple, Optional

# Determine device at the module level
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

class SentimentAnalyzer:
    """Encapsulates FinBERT model loading and sentiment analysis."""
    def __init__(self, model_name="ProsusAI/finbert"):
        """Loads the tokenizer and model upon initialization."""
        try:
            print(f"Initializing SentimentAnalyzer with model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                trust_remote_code=True 
            ).to(device)
            self.labels = ["positive", "negative", "neutral"]
            print("Tokenizer and model loaded successfully.")
        except Exception as e:
            print(f"Error loading model or tokenizer: {e}")
            self.tokenizer = None
            self.model = None
            self.labels = []
            raise

    def analyze(self, news: List[str]) -> Tuple[Optional[float], Optional[str]]:
        """
        Analyzes a list of news headlines and returns the overall sentiment
        and probability. Returns (None, None) if initialization failed or
        input is empty.
        """
        if not self.model or not self.tokenizer or not news:
            print("Analyzer not initialized properly or no news provided.")
            return 0.5, "neutral" 

        try:
            if isinstance(news, str):
                news = [news]
                
            tokens = self.tokenizer(news, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

            with torch.no_grad():
                result = self.model(tokens["input_ids"], attention_mask=tokens["attention_mask"])[
                    "logits"
                ]
            
            aggregated_logits = torch.sum(result, 0).cpu() 
            probabilities = torch.nn.functional.softmax(aggregated_logits, dim=-1)
            
            max_prob_index = torch.argmax(probabilities)
            probability = probabilities[max_prob_index].item()
            sentiment = self.labels[max_prob_index]
            
            if device == "cuda:0":
                del tokens
                del result
                torch.cuda.empty_cache()

            return probability, sentiment
            
        except Exception as e:
            print(f"Error during sentiment analysis: {e}")
            if device == "cuda:0":
                 if 'tokens' in locals(): del tokens
                 if 'result' in locals(): del result
                 torch.cuda.empty_cache()
            return 0.5, "neutral"

if __name__ == "__main__":
    print("Running FinBERT Utils test...")
    try:
        analyzer = SentimentAnalyzer()
        
        test_headlines = ["markets responded positively to the news!", "traders were pleased!"]
        probability, sentiment = analyzer.analyze(test_headlines)
        print(f"Test 1 Headlines: {test_headlines}")
        print(f"Sentiment: {sentiment}, Probability: {probability:.4f}\n")

        test_headlines_neg = ["earnings were lower than expected.", "investors sold off shares."]
        probability, sentiment = analyzer.analyze(test_headlines_neg)
        print(f"Test 2 Headlines: {test_headlines_neg}")
        print(f"Sentiment: {sentiment}, Probability: {probability:.4f}\n")
        
        test_headlines_single = ["The company announced a new product."]
        probability, sentiment = analyzer.analyze(test_headlines_single)
        print(f"Test 3 Headlines: {test_headlines_single}")
        print(f"Sentiment: {sentiment}, Probability: {probability:.4f}\n")

    except Exception as e:
        print(f"Error during test: {e}")

    print(f"CUDA available: {torch.cuda.is_available()}")
