from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

def evaluate_summary(text, reference_summary):
    model_inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(**model_inputs, max_length=150)
    generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    scores = scorer.score(reference_summary, generated_summary)
    return generated_summary, scores

# Example evaluation
text = "Company X reported an increase in revenue by 10% due to increased sales..."
reference_summary = "Company X saw a 10% revenue growth."
generated_summary, rouge_scores = evaluate_summary(text, reference_summary)
print("Generated Summary:", generated_summary)
print("ROUGE Scores:", rouge_scores)
