import torch
from src.train import get_dataloaders
from src.re_search import run_re
from src.plots import generate_re_plots

def main():
    # --- 1. Settings ---
    # Change sequentially for your testing: "cifar10", "cifar100", "tiny_imagenet"
    DATASET = "tiny_imagenet" 
    BATCH_SIZE = 128
    
    # Target 175 total models to force ~35-45 minutes of compute on a GPU
    # Tiny ImageNet will take longer because the images are 64x64 instead of 32x32.
    CYCLES = 50
    POPULATION_SIZE = 25
    SAMPLE_SIZE = 5
    EPOCHS_PER_MODEL = 3 
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 2. Setup Data ---
    print(f"Loading {DATASET}...")
    trainloader, testloader, num_classes = get_dataloaders(DATASET, BATCH_SIZE)

    # --- 3. Run Regularized Evolution ---
    _, saved_csv = run_re(
        cycles=CYCLES, 
        pop_size=POPULATION_SIZE, 
        sample_size=SAMPLE_SIZE, 
        trainloader=trainloader, 
        testloader=testloader, 
        num_classes=num_classes, 
        epochs=EPOCHS_PER_MODEL, 
        device=device,
        dataset_name=DATASET
    )

    # --- 4. Generate Plots ---
    print("\nGenerating visual comparisons based on Elapsed Time...")
    generate_re_plots(saved_csv, DATASET)

if __name__ == "__main__":
    main()