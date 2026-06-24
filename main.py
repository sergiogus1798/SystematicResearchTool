import Reader

stratsDirectory = "NAS100"

ReaderNAS = Reader.Reader(stratsDir=stratsDirectory)
ReaderNAS.readStrats(returns="Weekly")
