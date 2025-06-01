# Efficient_PLM_public

**Optimizing Protein Language Modeling via Constrained Learning**  
*Master’s Project — Duke Biostatistics Program (2023–2025)*

This project explores efficient pretraining of protein language models (PLMs) by actively identifying and sampling the most informative sequences. Our goal is to improve performance while reducing computational cost.

🚧 This repository is under active development!  
📄 A preprint on this work will be published soon!!

---

- Datasets
  - Uniref 20 to Uniref 50 
- Embedding Models
  - LSTM
  - Transformer
- Downstram Tasks
  - Protein Secondary Structure Prediction
  - Sub-Celluar Localization Prediction
  - AAV2 Capsid Protein VP-1 Prediction

---
To pre-train the Transformer model:
'''
python train_prose_masked_tf.py --clip 0.0001 --batch-size 100 --validate-every 10 --plr 5e-5 --cluster kmeans --epsilon 2.75 --nlayer 5 --d-model 1024 --name run1 --max-length 1024
'''

To run the pre-trained model on a specific downstream task:
'''
python ssp_tf.py --path-model saved_mdl_path 
'''
'''
python scl_tf.py --path-model saved_mdl_path 
'''
'''
python aav_tf.py --path-model saved_mdl_path 
'''
