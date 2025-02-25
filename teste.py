import torch
print(torch.cuda.is_available())  # Deve retornar True
print(torch.cuda.device_count())  # Deve mostrar o número de GPUs disponíveis
print(torch.cuda.get_device_name(0))  # Deve mostrar "NVIDIA GeForce RTX 3050"
print(torch.version.cuda) 