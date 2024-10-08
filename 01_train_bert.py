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
tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-128k-cased')
tokenizer.clean_up_tokenization_spaces = True

# Tokenize the text
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Split the dataset
tokenized_datasets = tokenized_datasets.shuffle(seed=42)
split_datasets = tokenized_datasets.train_test_split(test_size=0.2)

train_dataset = split_datasets['train']
eval_dataset = split_datasets['test']

# Set up the model
model = BertForSequenceClassification.from_pretrained('dbmdz/bert-base-turkish-128k-cased', num_labels=4)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=12,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=10,
    warmup_steps=10,
    weight_decay=0.015,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Custom save function to ensure tensors are contiguous
class ContiguousTrainer(Trainer):
    def save_model(self, output_dir: str, _internal_call: bool = False):
        # Make sure all model parameters are contiguous
        for param in self.model.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
        
        # Call the original save_model method
        super().save_model(output_dir, _internal_call)

# Set up the trainer with custom save function
trainer = ContiguousTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
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
test_sentence = "Bana bu eğitimin ana başlıklarını listele?"
inputs = tokenizer(test_sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
outputs = model(**inputs)
predicted_class = torch.argmax(outputs.logits).item()
print(f"Predicted class: {predicted_class}")