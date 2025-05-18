from .pastis import PASTISDataset

CLASS_NAMES = [
    'background', 'wheat', 'maize', 'rapeseed', 'barley', 'sunflower',
    'soybean', 'potato', 'sugar_beet', 'pea', 'oats', 'rye', 
    'grass', 'fallow', 'orchards', 'forest', 'urban', 'water', 'other'
]
NUM_CLASSES = len(CLASS_NAMES)


def get_dataset(dataset_name, **kwargs):
    if dataset_name == "pastis":
        return PASTISDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")