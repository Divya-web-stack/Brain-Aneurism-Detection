# Brain-Aneurism-Detection
🧠 Brain Aneurysm Detection & Segmentation with Multi-Task Learning
A deep learning framework for the automated detection and segmentation of intracranial aneurysms from 3D neuroimaging data using a multi-task Nested U-Net.


📖 Abstract
Intracranial aneurysms pose a significant neurological risk, with early and accurate detection being critical for preventing rupture. This project introduces a novel deep learning model, the NestedUnetWithClassifier, which performs two critical tasks simultaneously:

Classification: Identifies if an aneurysm is present in the scan.

Segmentation: Draws a precise pixel-level map of the aneurysm's location.

By leveraging a powerful Nested U-Net architecture, this framework provides a comprehensive, automated analysis of neuroimaging data, paving the way for faster and more accurate clinical diagnostics.

✨ Key Features
🧠 Multi-Task Learning: Simultaneously classifies and segments aneurysms in a single forward pass, creating an efficient and holistic model.

🚀 Advanced Architecture: Implements a Nested U-Net (UNet++), a state-of-the-art architecture known for its superior performance in medical image segmentation.

🔬 Slice-Level Analysis: Processes complex 3D DICOM and NIfTI volumes on a 2D slice-by-slice basis, making it trainable on consumer hardware.

▶️ End-to-End Pipeline: Includes complete scripts for training, validation, and prediction on new, unseen patient scans.

🔖 Checkpoint & Resuming: Training script automatically saves the best model and supports resuming from the last saved checkpoint, perfect for long training runs.

🏗️ Model Architecture
The core of this project is the NestedUnetWithClassifier. It enhances the standard UNet++ architecture by adding a classification head at the model's bottleneck.

Segmentation Path (The Artist 🎨): The full UNet++ decoder path produces a high-resolution segmentation mask.

Classification Path (The Detective 🕵️): A small series of fully connected layers are attached to the most compressed feature map (x4_0), which contains the richest high-level information about the entire image slice. This allows the model to make an accurate classification prediction.

🛠️ Setup and Installation
Get the project up and running on your local machine.

1. Clone the Repository
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

2. Create a Virtual Environment (Recommended)
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

3. Install Dependencies
This project requires the following libraries. Install them all with the provided requirements.txt file.

pip install -r requirements.txt

Key libraries include torch, pandas, pydicom, nibabel, and opencv-python.

📁 Dataset Structure
The model expects your data to be organized in the following folder structure:

your_main_data_folder/
├── series/
│   ├── PATIENT_UID_01/       <-- Folder with all .dcm files for one patient
│   │   ├── 1-01.dcm
│   │   ├── 1-02.dcm
│   │   └── ...
│   └── PATIENT_UID_02/
│
├── segmentations/
│   ├── PATIENT_UID_01.nii.gz <-- 3D segmentation mask
│   └── PATIENT_UID_02.nii.gz
│
├── final_train.csv           <-- CSV with classification labels
└── final_localize.csv        <-- CSV with localization info

🚀 How to Use
1. Training the Model
To start training from scratch, run the train.py script. The script will automatically save the best model and a log file in the models/ directory.

Make sure to set your data paths inside the train.py script first!

python train.py

2. Resuming Training
If your training was interrupted, you can resume from the last saved checkpoint. The script automatically saves a checkpoint.pth file in the model's output directory.

python train.py --resume "models/NestedUnetWithClassifier_multitask/checkpoint.pth"

3. Making Predictions
Use the predict.py script to run your trained model on a new, unseen patient scan.

python predict.py --model_path "models/NestedUnetWithClassifier_multitask/model.pth" --image_dir "C:/path/to/new/patient/dicom_folder/"
