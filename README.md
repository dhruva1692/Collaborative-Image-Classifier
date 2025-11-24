# Image Classification Project using PyTorch (Collaborative Work)

This repository contains a collaborative image classification project developed as part of our M.Tech coursework.  
The goal of the project was to build a working PyTorch pipeline for training a CNN model on a folder-based image dataset.  
The work includes dataset handling, preprocessing, model training, GPU usage, visualization, and evaluation.

This notebook was jointly developed with my classmates.  
I contributed mainly to the dataset preparation, transform pipeline, and training loop structure.

---

## Project Features

- Loads dataset from folder structure (multiple classes).
- Uses `torchvision.transforms` for:
  - Resize  
  - Normalization  
  - Data augmentation (optional)
- Custom `Dataset` class for image-label mapping.
- Training loop implemented from scratch using PyTorch.
- GPU support using:
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ```
- Tracks loss and accuracy during training.
- Shows visualizations:
  - Random sample images  
  - Class distribution  
  - Loss curves  
  - Accuracy curves  

---

##  Tools & Libraries

- PyTorch  
- Torchvision  
- Pillow  
- NumPy  
- Matplotlib  
- simple_image_download (for collecting images)


---

##  How to Run the Notebook

### 1. Install required packages
```bash
!pip install torch torchvision pillow matplotlib simple_image_download
```

### 2. (Optional) Mount Google Drive in Colab
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3. Set the dataset path  
Modify the path based on where your dataset is stored:

```python
dataset_path = "/content/drive/MyDrive/dataset_folder"
```

### 4. Run the notebook cells in order.

---

## 📂 Expected Dataset Structure

```
dataset/
│
├── class_1/
├── class_2/
├── class_3/
└── ...
```

The classes should be arranged in separate folders.

---

##  Collaboration

This project was completed as a team assignment.  
My contribution included:

- Setting up dataset directory structure  
- Writing initial version of the custom Dataset class  
- Adding image transformations and normalization  
- Helping structure the training loop  
- Organizing visualizations and accuracy tracking  

Other teammates contributed to model experimentation, dataset preparation, and testing.

---

##  Notes

- The dataset path in the notebook currently shows a Windows local path.  
  Update it as needed for Colab or Linux environments.
- The project can be extended by adding:
  - Confusion matrix  
  - Validation accuracy  
  - Model saving/loading  
  - More advanced CNN architectures  

---

##  Author

**Dhruv Kukadiya**  
M.Tech — AI & Data Science  
VIT Bhopal University

