import torch
DEVICE = 'cuda'

# def autobatch(model, imgsz, fraction=0.80):

#     device = torch.device("cuda")
#     model.to(DEVICE)

#     torch.cuda.empty_cache()

#     total_memory = torch.cuda.get_device_properties(0).total_memory
#     available_memory = fraction * total_memory

#     input_tensor = torch.randn(2, 3, imgsz, imgsz).to(device)

#     with torch.no_grad():
#         model(input_tensor)
#         memory_usage = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
#     max_batch_size = int(available_memory / (memory_usage)) // 4
#     if max_batch_size % 2 != 0:
#         max_batch_size += 1
#     if max_batch_size == 0:
#         max_batch_size = 2
    
#     return max_batch_size

def autobatch(model, imgsz, optimizer, fraction=0.7):
    device = torch.device("cuda")
    model.to(device)
    optimizer.zero_grad()
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, imgsz, imgsz).to(device)
    
    # Forward pass
    output = model(input_tensor)
    target = torch.randn_like(output)
    loss = torch.nn.functional.mse_loss(output, target)
    
    # Backward and optimizer step
    loss.backward()
    optimizer.step()
    
    # Calculate memory usage
    memory_used = torch.cuda.max_memory_allocated()
    memory_per_sample = memory_used / batch_size
    
    total_memory = torch.cuda.get_device_properties(device).total_memory
    max_batch_size = int((fraction * total_memory) / memory_per_sample)
    
    # Ensure even batch size and minimum 2

    if max_batch_size % 2 != 0:
        max_batch_size -= 1
        
    max_batch_size = max(max_batch_size, 4)

    model.zero_grad()
    torch.cuda.empty_cache()
    
    return max_batch_size