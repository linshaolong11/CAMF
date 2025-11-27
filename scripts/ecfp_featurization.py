#data
from rdkit import Chem
from rdkit.Chem import AllChem

def calculate_molecular_descriptors(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [0] * 1024
        return list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))
    except:
        return [0] * 1024 