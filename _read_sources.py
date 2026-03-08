import sys
files = [
    'src/models/hdim_model.py',
    'src/training/trainer.py', 
    'src/training/dataset.py'
]
for f in files:
    print(f'=== {f} ===')
    try:
        with open(f) as fh:
            print(fh.read())
    except Exception as e:
        print(f'ERROR: {e}')
    print()
