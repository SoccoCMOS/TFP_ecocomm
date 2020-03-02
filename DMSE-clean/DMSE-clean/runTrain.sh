rm -r model
rm -r summary
cp config.py backup/config.py
time python train.py
