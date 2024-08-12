import json
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
import torch

# Load the dataset from JSON
with open('bert_dataset.json', 'r') as f:
    data = json.load(f)

# Convert the JSON data to a format suitable for the Dataset class
dataset_dict = {
    'text': [item['text'] for item in data['questions']],
    'label': [item['label'] for item in data['questions']]
}

# Create a Dataset object
dataset = Dataset.from_dict(dataset_dict)

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the text
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Split the dataset
train_dataset = tokenized_datasets.shuffle(seed=42).select(range(8))  # Using 8 samples for training
eval_dataset = tokenized_datasets.shuffle(seed=42).select(range(8, 10))  # Using 2 samples for evaluation

# Set up the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Set up the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("./my_bert_model")
tokenizer.save_pretrained("./my_bert_model")

# Evaluate the model
eval_results = trainer.evaluate()
print(eval_results)

# Make predictions
test_sentence = "What software will we use in this AI course?"
inputs = tokenizer(test_sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
outputs = model(**inputs)
predicted_class = torch.argmax(outputs.logits).item()
print(f"Predicted class: {predicted_class}")