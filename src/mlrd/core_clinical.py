"""Opt-in clinical schema for the ADNI core-features experiment."""
from . import data

CORE_FEATURES = ['MMSE_MMSCORE', 'ADAS_TOTAL13', 'PTDEMOG_PTGENDER', 'AGE_derived']


def configure_core_clinical(features):
    if features != CORE_FEATURES:
        raise ValueError('Unexpected core clinical feature order')
    # Keep all imported references consistent; only this explicit entry point opts in.
    data.FEATURES[:] = features
    data.GENDER_INDEX = features.index('PTDEMOG_PTGENDER')
