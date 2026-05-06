from funasr import AutoModel
from pathlib import Path

directory = Path("/home/victor/DataSet/IEMOCAP")
wav_files = [str(f.absolute()) for f in directory.rglob("*.wav")]

model_id = "iic/emotion2vec_plus_large"
model = AutoModel(
    model=model_id,
    hub="ms",  # Set hub to ms for ModelScope or hf for HuggingFace
    disable_update=True
)

rec_result = model.generate(wav_files, output_dir="/home/victor/Github/DGFF-SER/emo2vec_large_features", granularity="frame")