import torch

print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))  # GPU 0

if torch.cuda.is_available():
    print("Current Device:", torch.cuda.current_device())

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs Available: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

if torch.cuda.is_available():
    print(torch.cuda.memory_summary(device=0, abbreviated=True))  # GPU 0
