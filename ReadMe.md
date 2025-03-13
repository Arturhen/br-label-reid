# BR Label Reid Interface

This repository provides a set of scripts and interfaces for processing and analyzing video data for re-identification tasks. Follow the steps below to use the repository effectively.

## Prerequisites

Ensure you have the following installed:
- Python >= 3.8
- Required Python packages (install using `requirements.txt` if available)

## Example Videos and Model

You can download example videos and the model for testing from the following link:

[Download Example Videos and Model](https://drive.google.com/drive/folders/177BKcy_j-tj7Affnuo8OIGnokhhF-c4J?usp=sharing)
 
## YouTube Demonstration

Watch a demonstration of how to use this repository on YouTube:

[YouTube Demonstration Video](https://www.youtube.com/watch?v=your-video-link) (TODO)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/br-label-reid.git
    cd br-label-reid/interface
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Step 1: Extract ROI from Videos

Use the `interface_extract_roi.py` script to select and extract regions of interest (ROI) from your input videos.

```bash
python scripts/interface_extract_roi.py
```

Follow the on-screen instructions to select the input video and specify the output file.

### Step 2: Process Videos with SORT

Use the `interface_SORT.py` script to process the extracted ROIs with the SORT tracker.

```bash
python scripts/interface_SORT.py
```

Add your video sources, specify the output directory, and start the processing.

### Step 3: Compare MBR Features

Use the `compare_mbr.py` script to compare features extracted from the images using the MBR model.

```bash
python scripts/compare_mbr.py
```

This script will generate a CSV file with similarity results.

#### Modifying `compare_mbr.py`

Before running the script, you need to update the paths to the folders containing the images to be compared and the model configuration and weights. Open `compare_mbr.py` and modify the following lines:

1. Change the path to the first folder:
    ```python
    detect_track = '/home/artur/Documents/br-label-reid/saida_ex/c001'  # Update this path
    ```

2. Change the path to the second folder:
    ```python
    detect_track2 = '/home/artur/Documents/br-label-reid/saida_ex/c002'  # Update this path
    ```

3. Change the path to the model configuration file:
    ```python
    extractor = FeatureExtractor(
        config_path='/home/artur/Documents/br-label-reid/model/config.yaml',  # Update this path
        weights_path='/home/artur/Documents/br-label-reid/model/best_mAP.pt',  # Update this path
        # device= # Optional
    )
    ```

### Step 4: Finish Dataset Preparation

Use the `finish_dataset.py` script to finalize the dataset preparation by combining images, renaming them, and splitting them into training and test sets.

```bash
python scripts/finish_dataset.py
```

Follow the on-screen instructions to select the input folders, CSV file, and output directory.

## Detailed Instructions

### Extract ROI from Videos

1. Run the script:
    ```bash
    python scripts/interface_extract_roi.py
    ```
2. Select the input video file.
3. Specify the output file name.
4. A window will open to select the region of interest (ROI) in the first frame of the video. Use the mouse to draw a rectangle around the ROI and press ENTER to confirm.
5. The script will process the video and save the extracted ROI to the specified output file.

### Process Videos with SORT

1. Run the script:
    ```bash
    python scripts/interface_SORT.py
    ```
2. Click "Adicionar Source" to add video sources or RTSP URLs.
3. Enter the camera ID for each source.
4. Select the base output directory.
5. Optionally, check "Manter apenas 3 imagens por ID" to keep only three images per ID.
6. Click "Iniciar Processamento" to start processing the videos.

### Compare MBR Features

1. Run the script:
    ```bash
    python scripts/compare_mbr.py
    ```
2. The script will extract features from images in the specified folders and compare them using the MBR model.
3. The results will be saved to `similarity_results.csv` and `similarity_results.txt`.

### Finish Dataset Preparation

1. Run the script:
    ```bash
    python scripts/finish_dataset.py
    ```
2. Select the input folders for the two sets of images.
3. Select the CSV file containing the comparison results.
4. Select the output directory.
5. The script will combine images, rename them, and split them into training and test sets. It will also split the test set into test and query sets.

## Acknowledgments

- [SORT: A Simple, Online and Realtime Tracker](https://github.com/abewley/sort)
- [MBR Model](https://github.com/videturfortuna/vehicle_reid_itsc2023)
