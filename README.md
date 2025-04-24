# NeuralComputing


```bash
python3 -m venv nc_venv
source nc_venv/bin/activate
pip install torchvision==0.21.0 torch==2.6.0 notebook==7.3.3 ipykernel==6.29.5
```

and
```bash
screen -S session_name
python model.py
Ctrl + A, then D
screen -r session_name
```