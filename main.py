from src.srt.loader import loader
from src.srt.overfitting import cscv

stratsDirectory = "NAS100"

ReaderNAS = loader.loader(stratsDir=stratsDirectory)
OriginalDataset = ReaderNAS.readStrats(returns="Weekly")

CSCV = cscv.cscv(OriginalDataset, S = 16)