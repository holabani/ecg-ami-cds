# Pretrained weights (optional to commit)

To let **clone-and-run** users skip training, add **one** of these files (same format as `train.py` outputs):

- `best_model_fold1.pt` (preferred)
- `best_model.pt`

After training locally:

```bash
cp /path/to/your/checkpoints/best_model_fold1.pt backend/checkpoints/best_model_fold1.pt
git add backend/checkpoints/best_model_fold1.pt
git commit -m "Add pretrained CardioSense checkpoint"
```

**Large files:** GitHub warns above ~50 MB and blocks files above 100 MB. Use [Git LFS](https://git-lfs.com/) (`git lfs track "*.pt"`) or attach the `.pt` to a **GitHub Release** and set `CARDIOSENSE_CHECKPOINT_URL` in `docker-compose.yml` to that asset URL so the backend downloads it on first start.
