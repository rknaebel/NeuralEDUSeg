cp -r /opt/data data
python scripts/run.py --prepare --rst_dir data
python scripts/run.py --train --rst_dir data --model_dir /build/models --restore
python app/api.py 0.0.0.0 --port 8080 --debug --model_dir /build/models