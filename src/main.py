import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch.optim import Adam
from sklearn.metrics.pairwise import cosine_similarity

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#------------------------Teacher and Student Model Selection-------------------------------

# Load teacher and student models and their tokenizers
teacher_model = AutoModel.from_pretrained("esm2_t33_650M_UR50D").to(device)
student_model = AutoModel.from_pretrained("esm2_t6_8M_UR50D").to(device)
teacher_tokenizer = AutoTokenizer.from_pretrained("esm2_t33_650M_UR50D")
student_tokenizer = AutoTokenizer.from_pretrained("esm2_t6_8M_UR50D")

# Set teacher model to evaluation mode
teacher_model.eval()

#------------------------Distillation Training Setup-------------------------------

# Define mean squared error loss for logits alignment
mse_loss = nn.MSELoss()

# Function to compute kernel (similarity matrix) for embeddings
def compute_kernel_matrix(embeddings):
    # Calculate cosine similarity between each pair of token embeddings
    return cosine_similarity(embeddings.cpu().detach().numpy())

# Function to calculate kernel alignment loss between teacher and student embeddings
def kernel_alignment_loss(teacher_embeddings, student_embeddings):
    teacher_kernel = compute_kernel_matrix(teacher_embeddings)
    student_kernel = compute_kernel_matrix(student_embeddings)
    # Calculate mean squared error between teacher and student similarity matrices
    return mse_loss(torch.tensor(teacher_kernel), torch.tensor(student_kernel))

# Combined distillation loss function
def compute_distillation_loss(teacher_logits, student_logits, teacher_embeddings, student_embeddings):
    logits_mse_loss = mse_loss(teacher_logits, student_logits)
    alignment_loss = kernel_alignment_loss(teacher_embeddings, student_embeddings)
    return logits_mse_loss + alignment_loss

# Initialize optimizer for student model
optimizer = Adam(student_model.parameters(), lr=1e-4)

#------------------------Training Loop-------------------------------

# Assuming `train_dataloader` is set up to load batches of sequences
epochs = 3  # Set number of epochs
for epoch in range(epochs):
    student_model.train()
    
    for batch in train_dataloader:
        # Load data and prepare it for model input
        sequences = batch['sequence']  # replace with actual key in your dataloader
        
        # Tokenize input sequences for both teacher and student models
        teacher_tokens = teacher_tokenizer(sequences, return_tensors="pt", padding=True, truncation=True).to(device)
        student_tokens = student_tokenizer(sequences, return_tensors="pt", padding=True, truncation=True).to(device)
        
        # Forward pass through teacher (no gradients needed)
        with torch.no_grad():
            teacher_outputs = teacher_model(**teacher_tokens, output_hidden_states=True)
            teacher_logits = teacher_outputs.logits
            teacher_embeddings = teacher_outputs.hidden_states[-1]  # Last layer's hidden states
        
        # Forward pass through student
        student_outputs = student_model(**student_tokens, output_hidden_states=True)
        student_logits = student_outputs.logits
        student_embeddings = student_outputs.hidden_states[-1]
        
        # Compute distillation loss
        loss = compute_distillation_loss(teacher_logits, student_logits, teacher_embeddings, student_embeddings)
        
        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")

#------------------------Evaluation-------------------------------

# Define function to evaluate student model against teacher model
def evaluate_model(test_dataloader):
    student_model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_dataloader:
            sequences = batch['sequence']
            teacher_tokens = teacher_tokenizer(sequences, return_tensors="pt", padding=True, truncation=True).to(device)
            student_tokens = student_tokenizer(sequences, return_tensors="pt", padding=True, truncation=True).to(device)
            
            # Forward pass for teacher and student models
            teacher_outputs = teacher_model(**teacher_tokens, output_hidden_states=True)
            teacher_logits = teacher_outputs.logits
            teacher_embeddings = teacher_outputs.hidden_states[-1]
            
            student_outputs = student_model(**student_tokens, output_hidden_states=True)
            student_logits = student_outputs.logits
            student_embeddings = student_outputs.hidden_states[-1]
            
            # Compute distillation loss for evaluation
            loss = compute_distillation_loss(teacher_logits, student_logits, teacher_embeddings, student_embeddings)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(test_dataloader)
    print(f"Average Evaluation Loss: {avg_loss}")

# Example call for evaluation
# evaluate_model(test_dataloader)
