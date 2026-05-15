import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# Select device GPU if available otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Load trained BERT model and tokenizer
model_path = r"C:\Users\ilsem\Documents\Thesis - memoire\dataset\bert_trained_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()


# Load dataset containing reviews
dataset_path = r"C:\Users\ilsem\Documents\Thesis - memoire\dataset\review_comment_message_original.csv"
df = pd.read_csv(dataset_path)


print("Total reviews loaded:", len(df))


# Extract review column and replace missing values
comments = df["review_comment_message"].fillna("").tolist()


# Tokenize text
encoded_inputs = tokenizer(
   comments,
   padding=True,
   truncation=True,
   max_length=128,
   return_tensors="pt"
)


# Create batches for faster inference
dataset = TensorDataset(encoded_inputs["input_ids"], encoded_inputs["attention_mask"])
dataloader = DataLoader(dataset, batch_size=32)


# Store predictions
all_preds = []


# Prediction loop with progress bar
for batch in tqdm(dataloader, desc="Predicting sentiment"):
   input_ids, attention_mask = [x.to(device) for x in batch]


   with torch.no_grad():
       outputs = model(input_ids=input_ids, attention_mask=attention_mask)
       preds = torch.argmax(outputs.logits, dim=1)


   all_preds.extend(preds.cpu().numpy())


# Convert numeric labels to sentiment names
label_mapping = {0: "negative", 1: "neutral", 2: "positive"}
pred_labels = [label_mapping[p] for p in all_preds]


# Create output dataframe with only needed columns
output_df = pd.DataFrame({
   "review": comments,
   "sentiment": pred_labels
})


# Print summary counts
print("\nSentiment counts")
print(output_df["sentiment"].value_counts())


# Save results automatically
output_path = r"C:\Users\ilsem\Documents\Thesis - memoire\dataset\review_comment_message_classified.csv"
output_df.to_csv(output_path, index=False)


print("\nFile saved successfully at:")
print(output_path)
