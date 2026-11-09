#core imports
import torch
from torch import optim
from torch import nn

#config imports
from config import DISPLAY_INFO, GAUSSIAN_BLUR_SIGMA

#model imports
from model import CNN, get_data_loader, misc_info
from config import SUPPORTED_KERNEL_SIZES, LEARNING_RATE, EPOCHS, SHOW_PROGRESS_BAR

#misc imports
import time
import os
import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

with open(f"results/loading_times.txt", "w") as f:
            f.write(f"Sigma\tTime\n")
            


for blur_sigma in GAUSSIAN_BLUR_SIGMA:
    #region Data loaders w/ Gaussian blur
    if blur_sigma < 0:
        raise ValueError("Gaussian blur sigma must be non-negative.")
    start_time = time.perf_counter()
    info = misc_info(display = DISPLAY_INFO)
    if blur_sigma == 0:
        blur = False
    else:
        blur = True
    train_loader = get_data_loader(split=0, blur=blur, sigma=blur_sigma)
    val_loader = get_data_loader(split=1, blur=blur, sigma=blur_sigma)
    test_loader = get_data_loader(split=2, blur=blur, sigma=blur_sigma)
    loader_time = time.perf_counter() - start_time
    print(f"Data loaders loaded in {loader_time} seconds for sigma={blur_sigma}.")
    with open(f"results/loading_times.txt", "a") as f:
            f.write(f"{blur_sigma}\t{loader_time}\n")
    #endregion    

    #running model training and evaluation
    for kernel_size_1 in SUPPORTED_KERNEL_SIZES:
        for kernel_size_2 in SUPPORTED_KERNEL_SIZES:
            for kernel_size_3 in SUPPORTED_KERNEL_SIZES:
                #region results and checkpoints
                folder_path = f"results/sigma_{blur_sigma}/ker_sizes_({kernel_size_1},{kernel_size_2},{kernel_size_3})"
                start_epoch = 0
                new_model = True
                if os.path.exists(os.path.join(folder_path, "results.txt")):
                    checkpoint = torch.load(os.path.join(folder_path, "model_checkpoint.pth"), weights_only=True)
                    start_epoch = checkpoint['epoch'] + 1
                    if start_epoch >= EPOCHS:  
                        print(f"Skipping model B:{blur_sigma} K:({kernel_size_1},{kernel_size_2},{kernel_size_3})...")
                        break
                    else:
                        print(f"Skipping epochs to {start_epoch+1}")
                        new_model = False
                else:
                    os.makedirs(folder_path, exist_ok=True)
                    results_file = os.path.join(folder_path, "results.txt")
                    checkpoint_file = os.path.join(folder_path, "model_checkpoint.pth")
                    with open(results_file, "w") as f:
                        f.write(f"Blur sigma: {blur_sigma}\tKernel sizes: ({kernel_size_1}, {kernel_size_2}, {kernel_size_3})\n")
                        f.write(f"Epoch\tTrain Accuracy\tVal Accuracy\tTest Accuracy\tTime Took\n")
                #endregion

                #region training and evaluation
                epoch_num = 1+start_epoch
                print(f"Model B:{blur_sigma} K:({kernel_size_1},{kernel_size_2},{kernel_size_3})-----------------")
                model = CNN(kernel_sizes=[kernel_size_1, kernel_size_2, kernel_size_3]).to(device)
                loss_fn = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
                if not new_model:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                for t in range(epoch_num, EPOCHS+1):
                    start_time = time.perf_counter()
                    train_accuracy = 0.0
                    val_accuracy = 0.0
                    test_accuracy = 0.0
                    if SHOW_PROGRESS_BAR:
                        train_loader = tqdm(train_loader, desc="Training")
                        val_loader = tqdm(val_loader, desc="Validation")
                        test_loader = tqdm(test_loader, desc="Testing")

                    #Training loop
                    model.train()
                    for batch in train_loader:
                        inputs, labels = batch
                        inputs, labels = inputs.to(device), labels.to(device)
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = loss_fn(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        _, predicted = torch.max(outputs.data, 1)
                        train_accuracy += (predicted == labels).sum().item()
                    train_accuracy = 100 * train_accuracy / len(train_loader.dataset)
                    
                    #Validation loop
                    model.eval()
                    with torch.no_grad():
                        for batch in val_loader:
                            inputs, labels = batch
                            inputs, labels = inputs.to(device), labels.to(device)
                            outputs = model(inputs)
                            _, predicted = torch.max(outputs.data, 1)
                            val_accuracy += (predicted == labels).sum().item()
                    val_accuracy = 100 * val_accuracy / len(val_loader.dataset)

                    # Test loop
                    with torch.no_grad():
                        for batch in test_loader:
                            inputs, labels = batch
                            inputs, labels = inputs.to(device), labels.to(device)
                            outputs = model(inputs)
                            _, predicted = torch.max(outputs.data, 1)
                            test_accuracy += (predicted == labels).sum().item()
                    test_accuracy = 100 * test_accuracy / len(test_loader.dataset)

                    epoch_time = time.perf_counter() - start_time
                    print(f"Epoch {t} || Train Accuracy: {train_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%, Time: {epoch_time:.2f}s")
                    
                    with open(os.path.join(folder_path, "results.txt"), "a") as f:
                        f.write(f"{t}\t{train_accuracy:.2f}\t{val_accuracy:.2f}\t{test_accuracy:.2f}\t{epoch_time:.2f}\n")
                    # Save checkpoint
                    torch.save({
                        'epoch': t,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, os.path.join(folder_path, "model_checkpoint.pth"))
                #endregion
                
