from .ViCoSClothDataset import ClothDataset
from .PreprocessorDataset import PreprocessorDataset

def get_dataset(name, dataset_opts, preprocessor=None):
    if name.lower() == "vicos-towel":
        dataset = ClothDataset(**dataset_opts)
    else:
        raise RuntimeError("Dataset {} not available".format(name))

    if preprocessor is not None:
        dataset = PreprocessorDataset(preprocessor, dataset)

    return dataset
