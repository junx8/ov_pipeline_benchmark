from anomalib.data import MVTecAD
datamodule = MVTecAD(root="./datasets/MVTecAD", category="bottle")
datamodule.prepare_data()