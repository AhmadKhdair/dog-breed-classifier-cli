# Dog Breed Classifier — CLI Image Classification App

A command-line Python application that uses pretrained CNN architectures (ResNet, AlexNet, VGG) to classify pet images and determine whether an image contains a dog and, if so, its breed. Built as part of Udacity's AI Programming with Python Nanodegree.

## What it does

- Classifies a folder of images using a pretrained CNN (`resnet`, `alexnet`, or `vgg`).
- Extracts the "true" label from each filename and compares it to the model's prediction.
- Determines whether the classifier correctly identified a dog vs. not-a-dog.
- Reports summary statistics: overall accuracy, percentage correctly classified as dogs, percentage correctly classified by breed.
- Also includes a small tool to classify a folder of custom `uploaded_images/`.

## Project structure

```
data/
├── check_images.py              # Main entry point
├── get_input_args.py            # Parses --dir, --arch, --dogfile CLI args
├── get_pet_labels.py            # Builds the ground-truth labels dict from filenames
├── classifier.py                # Wraps torchvision pretrained models (resnet/alexnet/vgg)
├── classify_images.py           # Runs the classifier over the image set
├── adjust_results4_isadog.py    # Flags whether each label/classification is "a dog"
├── calculates_results_stats.py  # Computes summary statistics
├── print_results.py             # Prints the final report
├── dognames.txt                 # Reference list of valid dog breed names
├── imagenet1000_clsid_to_human.txt  # ImageNet class ID → human-readable label map
├── pet_images/                  # Sample labeled test images
├── uploaded_images/             # Custom test images (cat, dog, coffee mug, etc.)
├── run_models_batch.sh          # Runs all three architectures back-to-back
└── *_pet-images.txt             # Captured output from each architecture's run
```

## Requirements

```
torch
torchvision
pillow
```

Install with:
```bash
pip install torch torchvision pillow
```

## Usage

```bash
cd data
python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
```

`--arch` accepts `resnet`, `alexnet`, or `vgg`.

Run all three architectures and save each report to a file:
```bash
sh run_models_batch.sh
```

Classify your own images:
```bash
python check_images.py --dir uploaded_images/ --arch resnet --dogfile dognames.txt
```

## Results

Pre-computed results for all three architectures are included (`alexnet_pet-images.txt`, `resnet_pet-images.txt`, `vgg_pet-images.txt`) from a full run against the sample `pet_images/` set.

## Author

Ahmad Khdair — Udacity AI Programming with Python Nanodegree
