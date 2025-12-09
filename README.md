# asr-tts
1. Step by step
``` sh
python -m venv myenv

source myenv/bin/activate

pip install -r requirements.txt

pyinstaller --onefile --name ai_mate --collect-all language_tags --collect-all kokoro --collect-all misaki --collect-all espeakng_loader --collect-all en_core_web_sm --collect-all spacy main.py
```

