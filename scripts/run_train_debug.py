import importlib.util
import os
import traceback
from types import SimpleNamespace

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
trainer_path = os.path.join(repo_root, 'train_depression_classifier.py')
spec = importlib.util.spec_from_file_location('train_depression_classifier', trainer_path)
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)

args = SimpleNamespace(
    model='roberta-base',
    data_path='data/dreaddit_sample.csv',
    test_size=0.2,
    epochs=1,
    batch_size=2,
    lr=2e-5,
    weight_decay=0.01,
    max_length=128,
    output_dir='models/trained',
    run_name='smoke_test_debug',
    seed=42,
    no_cuda=True
)

try:
    trainer.train_model(args)
except Exception as e:
    print('Exception raised during training:')
    traceback.print_exc()
    raise
