from ml.image.extractor.base import Extractor

from ml.common.extractor.tldr import TLDRExtractor

EXTRACTOR_REGISTRY = {
    Extractor.name: Extractor,
    TLDRExtractor.name: TLDRExtractor
}
