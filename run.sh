source ~/anaconda3/etc/profile.d/conda.sh
conda activate ail301
cd .. && uvicorn app.main:app --host 0.0.0.0 --reload