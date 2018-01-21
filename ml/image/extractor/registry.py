from ml.image.extractor.base import Extractor
from ml.image.extractor.tldr import TLDRExtractor

EXTRACTOR_REGISTRY = {
    Extractor.name: Extractor,
    TLDRExtractor.name: TLDRExtractor
}
