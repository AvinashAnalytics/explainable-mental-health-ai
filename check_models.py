"""Check what models are available locally"""
from transformers import AutoModel
import os
import glob

cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
print(f'\n📦 HuggingFace Cache: {cache_dir}\n')

if os.path.exists(cache_dir):
    models = glob.glob(os.path.join(cache_dir, 'models--*'))
    print(f'Cached models: {len(models)}\n')
    for m in models[:15]:
        model_name = os.path.basename(m).replace("models--", "").replace("--", "/")
        print(f'  ✅ {model_name}')
else:
    print('⚠️  No cache found - models will download on first use')

print('\n' + '='*60)
print('💡 Models available for training:')
print('='*60)
print('✅ distilbert-base-uncased (250MB)')
print('✅ bert-base-uncased (440MB)')
print('✅ roberta-base (500MB)')
print('\n⚠️  Custom models (may not exist):')
print('❓ mental/mental-bert-base-uncased')
print('❓ mental/mental-roberta-base')
print('\n💡 Recommendation: Use standard models (distilbert, bert, roberta)')
