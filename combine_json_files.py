import json
import codecs

# Initialize an empty list to store all questions
all_questions = []

# Iterate through the 5 input files
for i in range(4):
    filename = f"./data_preparation/data{i}.json"
    
    # Open and read each file with UTF-8 encoding
    with codecs.open(filename, 'r', 'utf-8') as file:
        data = json.load(file)
        
        # Extend the all_questions list with questions from this file
        all_questions.extend(data["questions"])

# Create the final structure
combined_data = {
    "questions": all_questions
}

# Write the combined data to bert_dataset.json with UTF-8 encoding
with codecs.open("bert_dataset.json", 'w', 'utf-8') as outfile:
    json.dump(combined_data, outfile, ensure_ascii=False, indent=2)

print("Combined JSON file 'bert_dataset.json' has been created successfully.")