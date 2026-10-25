#core imports
import torch

#config imports
from config import DISPLAY_INFO, GAUSSIAN_BLUR_SIGMA

#model imports
from model import CNN, get_data_loader, misc_info
from config import SUPPORTED_KERNEL_SIZES, LEARNING_RATE, BATCH_SIZE, EPOCHS, SHOW_PROGRESS_BAR

#misc imports
import time
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


for blur_sigma in GAUSSIAN_BLUR_SIGMA:
    #Data loaders w/ Gaussian blur
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
    print(f"Data loaders loaded in {time.perf_counter() - start_time:.2f} seconds for sigma={blur_sigma}.")

    #running model training and evaluation
    for kernel_size_1 in SUPPORTED_KERNEL_SIZES:
        for kernel_size_2 in SUPPORTED_KERNEL_SIZES:
            for kernel_size_3 in SUPPORTED_KERNEL_SIZES:
                #results and checkpoints
                folder_path = f"results/sigma_{blur_sigma}/ker_sizes_({kernel_size_1},{kernel_size_2},{kernel_size_3})"
                os.makedirs(folder_path, exist_ok=True)
                if os.path.exists(os.path.join(folder_path, "results.txt")):
                    print(f"Skipping model B_S:{blur_sigma} K_S:({kernel_size_1},{kernel_size_2},{kernel_size_3})...")
                else:
                    results_file = os.path.join(folder_path, "results.txt")
                    checkpoint_file = os.path.join(folder_path, "model_checkpoint.pth")
                    with open(results_file, "w") as f:
                        f.write(f"Blur sigma\tKernel sizes:\n")
                        f.write(f"{blur_sigma}\t{kernel_size_1}_{kernel_size_2}_{kernel_size_3}\n")
                        f.write(f"Epoch\tTrain Accuracy\tVal Accuracy\tTest Accuracy\n")
                model = CNN(kernel_sizes=[kernel_size_1, kernel_size_2, kernel_size_3]).to(device)