from ml.common.preparer.base import Preparer
from ml.common.preparer.tldr import TLDRPreparer

PREPARER_REGISTRY = {
    Preparer.name: Preparer,
    TLDRPreparer.name: TLDRPreparer
}
