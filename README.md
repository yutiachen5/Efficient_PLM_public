# Efficient_PLM_public

**Optimizing Protein Language Modeling via Constrained Learning**  
*Master’s Project — Duke Biostatistics Program (2023–2025)*

This project explores efficient pretraining of protein language models (PLMs) by actively identifying and sampling the most informative sequences. Our goal is to improve performance while reducing computational cost.

🚧 This repository is under active development!  
📄 A preprint on this work will be published soon!! A draft can be found at: [paper] (https://drive.google.com/file/d/1tIs26sp6EDkKNGxVXnXBef_Ud3kC5cAe/view?usp=drive_link) 


---

- Datasets
  - Uniref 20 to Uniref 50 
- Training Objective
  - MLM
- Embedding Models
  - LSTM
  - Transformer
- Downstram Tasks
  - Protein Secondary Structure Prediction
  - Sub-Celluar Localization Prediction
  - AAV2 Capsid Protein VP-1 Prediction

---
To pre-train the Transformer model:
```
cd Efficient_PLM
conda create -f environment.yml
python train_prose_masked_tf.py \
  --clip 0.0001 \
  --batch-size 100 \
  --validate-every 10 \
  --plr 5e-5 \
  -n 2000 \
  --cluster kmeans \
  --epsilon 2.6 \
  --nlayer 6 \
  --d-model 320 \
  --weight-decay 0.01 \
  --nhead 4 \
  --max-length 1024 \
  --alpha-slack 0.1 \
  --lr-slack 0.05 \
  --encoding RoPE
```

To run the pre-trained model on a specific downstream task:
```
python ssp_tf.py --path-model saved_mdl_path 
```
```
python scl_tf.py --path-model saved_mdl_path 
```
```
python aav_tf.py --path-model saved_mdl_path 
```
