# 🧑‍🦰 Age and Gender Estimation from Noisy Face Images
This project implements a deep learning model that predicts a person's age and gender from face images, even when the images contain noise.

<br>

### 📂 Project Structure
```
.
├── config/
│   └── config.yaml               # Configuration file for hyperparameters
│
├── models/
│   └── model.py                  # Model architecture 
│
├── runs/
│   ├── best.pth                  # Best model checkpoint
│   └── events.out.tfevents...    # TensorBoard event logs
│
├── utils/
│   ├── dataset.py                # Custom Dataset class
│   └── util.py                   # Utility functions
│
├── main.py                       
└── trainer.py                 
```

<br>
   
### Requirement
- pip install -r requirements.txt

<br>

### How to run
Edit config/config.yaml as needed, then run:
- python main.py
The script will automatically execute training, testing, or inference based on the mode field in config.yaml.
