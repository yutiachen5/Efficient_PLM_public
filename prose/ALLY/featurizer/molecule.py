from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from torch.nn import ModuleList
from torch.nn.functional import one_hot

import sys
sys.path.append("..")
# from utils import canonicalize, get_logger
from .base import Featurizer

class MorganFeaturizer(Featurizer):
    def __init__(
        self,
        shape: int = 2048,
        radius: int = 2,
        save_dir: Path = Path().absolute(),
    ):
        super().__init__("Morgan", shape, save_dir)

        self._radius = radius

    def canonicalize(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, isomericSmiles=True)
        else:
            return None

    def smiles_to_morgan(self, smile: str):
        """
        Convert smiles into Morgan Fingerprint.
        :param smile: SMILES string
        :type smile: str
        :return: Morgan fingerprint
        :rtype: np.ndarray
        """
        try:
            smile = self.canonicalize(smile)
            mol = Chem.MolFromSmiles(smile)
            features_vec = AllChem.GetMorganFingerprintAsBitVect(
                mol, self._radius, nBits=self.shape
            )
            features = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(features_vec, features)
        except Exception as e:
            print(
                f"rdkit not found this smiles for morgan: {smile} convert to all 0 features"
            )
            features = np.zeros((self.shape,))
        return features

    def transform(self, smile: str) -> torch.Tensor:
        feats = torch.from_numpy(self.smiles_to_morgan(smile)).squeeze().float()
        if feats.shape[0] != self.shape:
            logg.warning("Failed to featurize: appending zero vector")
            feats = torch.zeros(self.shape)
        return feats