# CLASS_NAMES = ["Background", "Meadow", "Soft winter wheat", "Corn", "Winter barley", "Winter Rapeseed", "Spring barley", "Sunflower", "Grapewine", "Beet", "Winter triticale", "Winter durum wheat", "Fruits, vegetables, flowers", "Potatoes", "Leguminous fodder", "Soybeans", "Orchard", "Mixed cereal", "Sorghum"]
# NUM_CLASSES = 19  # 18 crops + background

from .pastis import PASTISDataset

CLASS_NAMES = [
    'background', 'wheat', 'maize', 'rapeseed', 'barley', 'sunflower',
    'soybean', 'potato', 'sugar_beet', 'pea', 'oats', 'rye', 
    'grass', 'fallow', 'orchards', 'forest', 'urban', 'water', 'other'
]
NUM_CLASSES = len(CLASS_NAMES)  # 19 (18 crops + background)

def get_dataset(dataset_name, **kwargs):
    if dataset_name == 'pastis':
        return PASTISDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")