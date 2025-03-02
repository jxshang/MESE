# MESE: Mining Emotional and Semantic Evolution from User Comments for Fake News Detection

#### Authors: Jiaxing Shang, Rong Xu, Mengya Guan, et al.

This project contains the source code for the work: **MESE: Mining Emotional and Semantic Evolution from User Comments for Fake News Detection**

------

# Table of Contents

- Data Preprocessing (pre_process folder):
  - `get_embedding.py`: Obtains initial feature representations of news and comment texts.
  - `get_emotion.py`: Extracts emotional embeddings from user comments.
  - `cat.py`: Combines the text embeddings and emotional embeddings into a dictionary.
- **data_input.py**: Loads the data.
- **model.py**: The overall model architecture.
- **run.py**: Main procedure for model training and testing.
- **setting.py**: Contains the parameter settings for the MESE model.

------

# Execution

### Data Preprocessing Phase:

1. **Obtain initial feature representations for news and comment texts**:
   
   ```bash
   cd pre_process
   python get_embedding.py
   ```

2. **Extract emotional embeddings from user comments**:
   
   - For the Weibo-comp dataset:
     
     ```bash
     python get_emotion.py
     ```
   
   - For the RumourEval-19 dataset:
     
     ```bash
     cd dependencies/twitter-emotion-recognition-master
     python get_emotion.py
     ```

3. **Combine the text embeddings and emotional embeddings into a single dictionary**:
   
   ```bash
   cd pre_process
   python cat.py
   ```

------

### Model Execution Phase:

```bash
python run.py
```

------

# Requirements

### Data Preprocessing Phase:

**get_embedding:**

- `numpy==1.19.5`
- `torch==1.4.0`
- `torchvision==0.9.1+cu101`
- `transformers==4.19.2`

**get_emotion:**

- `Keras==2.4.3`
- `pandas==0.24.1`
- `requests==2.31.0`
- `tqdm==4.62.3`

------

### Model Execution Phase:

- `numpy==1.19.5`

- `scikit-learn==1.0.2`

- `torchvision==0.9.1+cu101`

- `torch==1.4.0`

- `tqdm==4.62.3`

------

## Contact

+ shangjx@cqu.edu.cn (Jiaxing Shang)

+ rongxu126@gmail.com(Rong Xu)
