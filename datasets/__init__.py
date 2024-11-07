from .ViCoSClothDataset import ClothDataset
from .MuJoCoDataset import MuJoCoDataset
from .PreprocessorDataset import KeypointPreprocessorDataset

def get_dataset(name, dataset_opts, preprocessor=None):
    if name.lower() == "concat":
        from torch.utils.data import ConcatDataset
        dataset = ConcatDataset([get_dataset(db['name'], db['kwargs']) for db in dataset_opts])

    elif name.lower() == "vicos-towel":
        dataset = ClothDataset(**dataset_opts)
    elif name.lower() == "mujoco":
        dataset = MuJoCoDataset(**dataset_opts)
    else:
        raise RuntimeError("Dataset {} not available".format(name))

    if preprocessor is not None:
        dataset = KeypointPreprocessorDataset(preprocessor, dataset)

    return dataset
