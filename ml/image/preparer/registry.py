from ml.image.preparer.base import Preparer
from ml.image.preparer.tldr import TLDRPreparer

PREPARER_REGISTRY = {
    Preparer.name: Preparer,
    TLDRPreparer.name: TLDRPreparer
}
