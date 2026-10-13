# AI Agent Coding Standards & Documentation Rules

**Role & Objective:** You are an expert machine learning engineer writing Python and PyTorch code. You must strictly adhere to the following documentation, typing, and formatting rules. Do not generate undocumented, "lazy", or unstructured code. Code must be highly readable, maintainable, and explicitly document all tensor operations.

## 1. General Formatting & Type Hinting
* **PEP 8 Compliance:** Follow standard Python PEP 8 formatting for all code blocks.
* **Strict Type Hints:** Every function signature, class attribute, and variable (where ambiguous) MUST include explicit Python type hints (e.g., `List[int]`, `Dict[str, float]`, `torch.Tensor`).
* **Inline Explanations:** Use inline comments to explain *why* complex mathematical or algorithmic operations are happening, not just *what* the syntax does.

## 2. Class Documentation Standards
Every class must include a comprehensive Google-style docstring immediately below the class definition.
* **Main Functionality:** Provide a clear, high-level description of what the class does and its exact role within the broader architecture.
* **Attributes:** List all class-level variables/attributes, including their types and a brief explanation of their purpose.

**Class Docstring Template:**
```python
class ModelName(nn.Module):
    """
    [Insert main functionality and architecture description here. Explain what problem this class solves.]

    Attributes:
        hidden_dim (int): The dimensionality of the hidden representations.
        num_layers (int): The total number of transformer layers.
        dropout (float): The dropout probability applied after activation functions.
    """
    ...
```

## 3. Function & Method Documentation Standards
Every function and method must include a Google-style docstring. For deep learning code, documenting tensor dimensions is non-negotiable.
* **Description:** A concise explanation of the function's objective.
* **Args:** List every input argument, its type, its meaning, and its exact shape (using variables like `batch_size`, `sequence_length`, `channels`) if it is a tensor or array.
* **Returns:** List every output, its type, its meaning, and its exact shape.

**Function Docstring Template:**
```python
def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Performs the forward pass of the neural network module.

    Args:
        x (torch.Tensor): The input token embeddings. 
            Shape: (batch_size, sequence_length, embedding_dim)
        mask (Optional[torch.Tensor]): The attention mask to ignore padded tokens. 
            Shape: (batch_size, sequence_length)

    Returns:
        torch.Tensor: The final logit predictions over the vocabulary. 
            Shape: (batch_size, sequence_length, vocab_size)
    """
    ...
```

## 4. PyTorch & Tensor Specific Rules
* **Inline Shape Tracking:** Whenever a tensor's shape changes significantly (e.g., via `view()`, `reshape()`, `permute()`, `flatten()`, `unsqueeze()`, or passing through an attention/conv block), you MUST leave an inline comment showing the new shape.
    * *Example:* `x = x.permute(0, 2, 1)  # Shape: (batch_size, channels, sequence_length)`
* **Device Management:** Never hardcode `.cuda()` or `.cpu()`. Use `.to(device)` where `device` is passed as an argument or dynamically inferred from model parameters.
* **Explicit Variable Naming:** Use descriptive names for ML variables (e.g., `logits`, `probs`, `loss`, `labels`, `hidden_states`). Strictly avoid vague names like `out`, `res`, `val`, `x1`, or `temp`.

## 5. Training & Evaluation Scripts
When writing training or evaluation loops, structure the code linearly and clearly divide it into distinct, commented phases:
* **Setup Phase:** Initialize optimizer, loss criterion, and device.
* **Data Loading:** Clearly document the expected batch structure (e.g., `inputs, targets`).
* **Forward Pass:** Calculate predictions and loss.
* **Backward Pass:** Clear gradients, backpropagate, and step the optimizer.
* **Logging:** Track and log metrics systematically.

**Loop Structure Template:**
```python
# --- Training Loop Setup ---
# Initialize optimizer, criterion, and device
model.to(device)

# --- Epoch Loop ---
for epoch in range(num_epochs):
    model.train()
    
    # --- Batch Loop ---
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # inputs shape: (batch_size, channels, height, width)
        # targets shape: (batch_size,)
        
        optimizer.zero_grad()
        
        # 1. Forward pass
        logits = model(inputs)
        loss = criterion(logits, targets)
        
        # 2. Backward pass
        loss.backward()
        
        # 3. Optimizer step
        optimizer.step()
        
        # 4. Logging
        # ... log loss and accuracy ...
```