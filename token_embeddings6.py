from transformers import AutoTokenizer, AutoModel
import torch

#1 Load model and tokenizer with hidden states
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

#2 Input sentences
sentences = [
    "The wheather in Miami is better than Chicago.", 
    "Soccer is the most entertaining sport to watch.", 
    "Novak Djokovic is the best tennis player of all time."
]

#3 Tokenize sentences

tokens = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
input_ids = tokens["input_ids"]
attention_mask = tokens["attention_mask"] 

print("Token IDs for each sentence:")
for idx, sentence_ids in enumerate(input_ids):
    print(f"Sentence {idx+1}: {sentence_ids.tolist()}")

#4 Pass through the model to get all hidden states

with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    all_hidden_states = outputs.hidden_states

#5 Print first 10 dimensions of the first 10 tokens for transformer layers 1-5
print("\n=== Hidden States for Transformer Layers 1 to 5 (First 10 Tokens of First Sentence) ===")

tokens_str_first_sentence = tokenizer.convert_ids_to_tokens(input_ids[0])


for layer_idx in range(1, 6): 
    print(f"\n--- Layer {layer_idx} ---")
    
    current_layer_output = all_hidden_states[layer_idx]
    

    for token_pos in range(min(10, current_layer_output.size(1))):

        token_vector_slice = current_layer_output[0, token_pos, :10]
        
        values = [round(val.item(), 5) for val in token_vector_slice]
        
        token_string = tokens_str_first_sentence[token_pos]
        
        print(f"  Token {token_pos+1} ('{token_string}'): {values}")

        import torch.nn.functional as F

#Number of tokens in the first sentence
num_tokens = input_ids[0].size(0)  #Which gives the number of tokens

#Container - hold pairwise distance matrices
num_layers = 5
pairwise_distances = torch.zeros((num_layers, num_tokens, num_tokens))

#Compute cosine distances for each layer
for layer_idx in range(1, 6):  #Layers 1 - 5
    current_hidden = all_hidden_states[layer_idx][0]  # shape: (tokens, hidden_dim)
    
    #Normalizing each token vector
    normed = F.normalize(current_hidden, p=2, dim=1)
    
    #Computing cosine similarity matrix (tokens x tokens)
    cosine_sim = torch.matmul(normed, normed.T)
    
    #Converting to cosine distance
    cosine_dist = 1 - cosine_sim  #Cosine distance =is 1 - cosine similarity
    
    pairwise_distances[layer_idx - 1] = cosine_dist

# Printing 5x5 submatrix from each layer
print("\n=== First 5x5 Cosine Distance Submatrices ===")
for i in range(num_layers):
    print(f"\n--- Layer {i+1} ---")
    print(pairwise_distances[i, :5, :5])
