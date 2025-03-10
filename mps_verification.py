import torch

# Step 2: Verify MPS Availability
print("MPS Available:", torch.backends.mps.is_available())  # Should print True
print("MPS Built:", torch.backends.mps.is_built())          # Should print True

# Step 3: Move Tensors to MPS
device = torch.device("mps")  # Use Metal GPU

# Example tensor creation and operation
x = torch.randn(3, 3).to(device)
y = torch.randn(3, 3).to(device)
z = x @ y  # Matrix multiplication on MPS device

print("Matrix multiplication result on MPS:")
print(z) 