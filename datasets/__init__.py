from .ViCoSClothDataset import ClothDataset
from .PreprocessorDataset import KeypointPreprocessorDataset

def get_dataset(name, dataset_opts, preprocessor=None):
    if name.lower() == "vicos-towel":
        dataset = ClothDataset(**dataset_opts)
    else:
        raise RuntimeError("Dataset {} not available".format(name))

    if preprocessor is not None:
        dataset = KeypointPreprocessorDataset(preprocessor, dataset)

    return dataset
