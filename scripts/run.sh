cp -r /opt/data data
python scripts/run.py --prepare --rst_dir data
python scripts/run.py --train --rst_dir data --model_dir /opt/models