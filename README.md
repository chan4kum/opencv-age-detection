# Age Detection

Estimates an age range for each detected face using a pretrained Caffe deep-learning model (Levi & Hassner).

Part of a series of beginner-friendly OpenCV projects.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python download_models.py     # one-time
python main.py                 # webcam
python main.py --image photo.jpg
```

Press `q` to quit any live window. Omit input arguments to use your webcam where supported.

## License

MIT
