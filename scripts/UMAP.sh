python UMAP.py \
    --emotion2vec_dir ./Features/IEMOCAP/emo2vec_large_features \
    --hubert_dir ./Features/IEMOCAP/hubert_large_features \
    --model_path ./models/fold1/best_model/pytorch_model.bin \
    --csv_path ./csv_files/IEMOCAP.csv \
    --output_dir ./results \
    --n_neighbors 10 \
    --min_dist 0.3