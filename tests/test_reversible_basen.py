"""Tests for the reversible, integrity-protected Base-N transport."""

import json

import numpy as np
import pandas as pd
import pytest
from category_encoders import BaseNEncoder, ReversibleBaseN
from category_encoders.reversible_basen import (
    MissingRowsError,
    OutOfVocabularyError,
    SchemaMismatchError,
    SourceMissingError,
    UnrecoverableCorruptionError,
    UnrecoverableMissingError,
    VocabularyIntegrityError,
    VocabularyVersionMismatchError,
)


def make_frame(n=120, ncat=6, seed=0, with_missing=True):
    """Build frame."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            'biz_id': [f'ID-{i:04d}' for i in range(n)],
            'city': rng.choice([f'c{i}' for i in range(ncat)], n),
            'tier': rng.choice(['x', 'y', 'z'], n),
        }
    )
    if with_missing:
        # NaN is present at fit time, so it becomes a genuine fitted category
        df.loc[rng.choice(n, 5, replace=False), 'city'] = np.nan
    return df


def make_frame_with_unseen_nan(n=120, ncat=6, seed=0):
    """Frame whose encoder never sees NaN, so a later NaN is a true missing."""
    rng = np.random.default_rng(seed)
    clean = pd.DataFrame(
        {
            'biz_id': [f'ID-{i:04d}' for i in range(n)],
            'city': rng.choice([f'c{i}' for i in range(ncat)], n),
            'tier': rng.choice(['x', 'y', 'z'], n),
        }
    )
    return clean


def make_encoder(df, base=2):
    """Build encoder."""
    return BaseNEncoder(cols=['city', 'tier'], base=base).fit(df)


def assert_roundtrip_equals(result, original, fields=('city', 'tier')):
    """Assert roundtrip equals."""
    merged = original[['biz_id', *fields]].merge(result, on='biz_id', suffixes=('_orig', '_back'))
    assert len(merged) == len(original)
    for field in fields:
        equal = (merged[f'{field}_orig'] == merged[f'{field}_back']) | (
            merged[f'{field}_orig'].isna() & merged[f'{field}_back'].isna()
        )
        assert equal.all(), f'field {field} did not round-trip'


# ---------------------------------------------------------------------------
# clean round trips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('base', [1, 2, 3, 5])
def test_clean_roundtrip_several_bases(base):
    """Clean roundtrip several bases."""
    df = make_frame()
    rbn = ReversibleBaseN(make_encoder(df, base=base), 'biz_id')
    transport, manifest = rbn.export_frame(df)
    result = rbn.import_frame(transport, manifest, on_violation='report')
    assert_roundtrip_equals(result, df)


def test_strict_mode_succeeds_on_fully_clean_data():
    """Strict mode succeeds on fully clean data."""
    df = make_frame(with_missing=False)
    rbn = ReversibleBaseN(make_encoder(df), 'biz_id')
    transport, manifest = rbn.export_frame(df)
    result = rbn.import_frame(transport, manifest)  # default: raise
    assert_roundtrip_equals(result, df)


def test_transport_carries_exactly_two_redundancy_columns_per_field():
    """Transport carries exactly two redundancy columns per field."""
    df = make_frame()
    rbn = ReversibleBaseN(make_encoder(df), 'biz_id')
    transport, manifest = rbn.export_frame(df)
    for field, fmeta in manifest['fields'].items():
        assert fmeta['p0_col'] in transport.columns
        assert fmeta['p1_col'] in transport.columns
        # exactly the two redundancy symbols, no more
        redundancy = [c for c in transport.columns if c.startswith(field) and ('__p' in c)]
        assert sorted(redundancy) == sorted([fmeta['p0_col'], fmeta['p1_col']])


def test_file_roundtrip(tmp_path):
    """File roundtrip."""
    df = make_frame()
    rbn = ReversibleBaseN(make_encoder(df), 'biz_id')
    data = tmp_path / 'transport.csv'
    manifest_file = tmp_path / 'manifest.json'
    rbn.to_files(df, data, manifest_file)
    result = rbn.from_files(data, manifest_file, on_violation='report')
    assert_roundtrip_equals(result, df)


def test_integer_row_ids_and_categorical_dtype_roundtrip():
    """Integer row ids and categorical dtype roundtrip."""
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            'rid': np.arange(1000, 1030),
            'bucket': pd.Categorical(rng.choice(['p', 'q', 'r'], 30)),
        }
    )
    encoder = BaseNEncoder(cols=['bucket'], base=2).fit(df)
    rbn = ReversibleBaseN(encoder, 'rid')
    transport, manifest = rbn.export_frame(df)
    result = rbn.import_frame(transport, manifest)
    assert result['bucket'].astype(str).tolist() == df['bucket'].astype(str).tolist()
    # row correspondence is preserved through the integer business id
    assert result['rid'].tolist() == df['rid'].tolist()


def test_nan_is_a_fitted_category_and_roundtrips():
    """Nan is a fitted category and roundtrips."""
    df = pd.DataFrame({'id': range(6), 'cat': ['a', 'b', np.nan, 'a', 'b', np.nan]})
    encoder = BaseNEncoder(cols=['cat'], base=2).fit(df)
    rbn = ReversibleBaseN(encoder, 'id')
    transport, manifest = rbn.export_frame(df)
    # a NaN seen at fit time is a genuine category -> status ok, digits present
    nan_rows = transport[transport['id'].isin([2, 5])]
    assert (nan_rows['cat__status'] == 'ok').all()
    result = rbn.import_frame(transport, manifest)
    assert result['cat'].isna().tolist() == [False, False, True, False, False, True]


# ---------------------------------------------------------------------------
# single wrong value at an unknown position
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('base', [2, 3, 5])
def test_single_unknown_position_error_is_corrected(base):
    """Single unknown position error is corrected."""
    df = make_frame(with_missing=False)
    rbn = ReversibleBaseN(make_encoder(df, base=base), 'biz_id')
    transport, manifest = rbn.export_frame(df)
    digit_cols = manifest['fields']['city']['digit_cols']
    target = 7
    col = digit_cols[0]
    transport.at[target, col] = (transport.at[target, col] + 1) % base
    result = rbn.import_frame(transport, manifest)
    assert result.loc[target, 'city'] == df.loc[target, 'city']


def test_exhaustive_single_error_sweep():
    """Exhaustive single error sweep."""
    df = make_frame(n=60, ncat=12, seed=11, with_missing=False)
    rbn = ReversibleBaseN(make_encoder(df), 'biz_id')
    transport, manifest = rbn.export_frame(df)
    fmeta = manifest['fields']['city']
    base, digit_cols = fmeta['base'], fmeta['digit_cols']
    for idx in range(20):
        for col in digit_cols:
            for wrong in range(base):
                if wrong == transport.at[idx, col]:
                    continue
                trial = transport.copy()
                trial.at[idx, col] = wrong
                result = rbn.import_frame(trial, manifest)
                assert result.loc[idx, 'city'] == df.loc[idx, 'city']


# ---------------------------------------------------------------------------
# up to two missing values at known positions
# ---------------------------------------------------------------------------


def test_one_erasure_is_restored():
    """One erasure is restored."""
    df = make_frame(with_missing=False)
    rbn = ReversibleBaseN(make_encoder(df), 'biz_id')
    transport, manifest = rbn.export_frame(df)
    transport.at[9, 'city_1'] = np.nan
    result = rbn.import_frame(transport, manifest)
    assert result.loc[9, 'city'] == df.loc[9, 'city']


def test_two_erasures_are_restored():
    """Two erasures are restored."""
    df = make_frame(with_missing=False)
    rbn = ReversibleBaseN(make_encoder(df), 'biz_id')
    transport, manifest = rbn.export_frame(df)
    cols = manifest['fields']['city']['digit_cols']
    transport.loc[9, cols[:2]] = np.nan
    result = rbn.import_frame(transport, manifest)
    assert result.loc[9, 'city'] == df.loc[9, 'city']


def test_exhaustive_two_erasure_sweep():
    """Exhaustive two erasure sweep."""
    df = make_frame(n=60, ncat=12, seed=12, with_missing=False)
    rbn = ReversibleBaseN(make_encoder(df), 'biz_id')
    transport, manifest = rbn.export_frame(df)
    digit_cols = manifest['fields']['city']['digit_cols']
    for idx in range(15):
        for a in range(len(digit_cols)):
            for b in range(a + 1, len(digit_cols)):
                trial = transport.copy()
                trial.loc[idx, [digit_cols[a], digit_cols[b]]] = np.nan
                result = rbn.import_frame(trial, manifest)
                assert result.loc[idx, 'city'] == df.loc[idx, 'city']


def test_three_missing_cells_raise_typed_error():
    """Three missing cells raise typed error."""
    df = make_frame(ncat=10, with_missing=False)
    rbn = ReversibleBaseN(make_encoder(df), 'biz_id')
    transport, manifest = rbn.export_frame(df)
    cols = manifest['fields']['city']['digit_cols']
    assert len(cols) >= 3
    transport.loc[3, cols] = np.nan
    with pytest.raises(UnrecoverableMissingError):
        rbn.import_frame(transport, manifest)


# ---------------------------------------------------------------------------
# damaged redundancy never decodes silently
# ---------------------------------------------------------------------------


def test_missing_redundancy_symbol_raises():
    """Missing redundancy symbol raises."""
    df = make_frame(with_missing=False)
    rbn = ReversibleBaseN(make_encoder(df), 'biz_id')
    transport, manifest = rbn.export_frame(df)
    transport.at[4, 'city__p0'] = np.nan
    with pytest.raises(UnrecoverableCorruptionError):
        rbn.import_frame(transport, manifest)


def test_wrong_redundancy_symbol_raises():
    """Wrong redundancy symbol raises."""
    df = make_frame(with_missing=False)
    rbn = ReversibleBaseN(make_encoder(df), 'biz_id')
    transport, manifest = rbn.export_frame(df)
    transport.at[4, 'city__p1'] = transport.at[4, 'city__p1'] + 1
    with pytest.raises(UnrecoverableCorruptionError):
        rbn.import_frame(transport, manifest)


def test_double_error_outside_guarantee_raises_not_miscorrects():
    """Double error outside guarantee raises not miscorrects."""
    df = make_frame(with_missing=False)
    rbn = ReversibleBaseN(make_encoder(df), 'biz_id')
    transport, manifest = rbn.export_frame(df)
    for col in ['city_0', 'city_1']:
        transport.at[5, col] = (transport.at[5, col] + 1) % 2
    with pytest.raises(UnrecoverableCorruptionError):
        rbn.import_frame(transport, manifest)


def test_corrupted_row_is_never_decoded_into_another_category():
    """Corrupted row is never decoded into another category."""
    df = make_frame(with_missing=False)
    rbn = ReversibleBaseN(make_encoder(df), 'biz_id')
    transport, manifest = rbn.export_frame(df)
    transport.at[6, 'city__p0'] = np.nan  # unrecoverable group
    result = rbn.import_frame(transport, manifest, on_violation='report')
    assert pd.isna(result.loc[6, 'city'])
    assert result.loc[6, 'city__status'] == 'corruption'
    # the unaffected field on the same row is still independently recovered
    assert result.loc[6, 'tier'] == df.loc[6, 'tier']


# ---------------------------------------------------------------------------
# the three conditions stay distinct
# ---------------------------------------------------------------------------


def test_missing_vs_oov_are_distinct_statuses():
    """Missing vs oov are distinct statuses."""
    # fit without NaN so a later NaN is a genuine source-level missing value
    clean = make_frame_with_unseen_nan()
    rbn = ReversibleBaseN(make_encoder(clean), 'biz_id')
    df = clean.copy()
    df.loc[0, 'city'] = np.nan  # unseen NaN -> missing
    df.loc[1, 'city'] = 'planet_mars'  # unseen category -> oov
    transport, manifest = rbn.export_frame(df)
    assert transport.loc[0, 'city__status'] == 'missing'
    assert transport.loc[1, 'city__status'] == 'oov'
    assert transport.loc[1, 'city__raw'] == 'planet_mars'
    # neither row carries redundancy
    assert pd.isna(transport.loc[0, 'city__p0'])
    assert pd.isna(transport.loc[1, 'city__p0'])

    result = rbn.import_frame(transport, manifest, on_violation='report')
    assert result.loc[0, 'city__status'] == 'missing'
    assert result.loc[1, 'city__status'] == 'oov'
    assert pd.isna(result.loc[0, 'city']) and pd.isna(result.loc[1, 'city'])


def test_missing_raises_source_missing_error():
    """Missing raises source missing error."""
    clean = make_frame_with_unseen_nan()
    rbn = ReversibleBaseN(make_encoder(clean), 'biz_id')
    df = clean.copy()
    df.loc[2, 'city'] = np.nan
    transport, manifest = rbn.export_frame(df)
    with pytest.raises(SourceMissingError):
        rbn.import_frame(transport, manifest)


def test_oov_raises_out_of_vocabulary_error():
    """Oov raises out of vocabulary error."""
    clean = make_frame_with_unseen_nan()
    rbn = ReversibleBaseN(make_encoder(clean), 'biz_id')
    df = clean.copy()
    df.loc[1, 'city'] = 'planet_mars'
    transport, manifest = rbn.export_frame(df)
    transport = transport[transport['city__status'] != 'missing'].reset_index(drop=True)
    with pytest.raises(OutOfVocabularyError):
        rbn.import_frame(transport, manifest)


def test_fitted_nan_category_roundtrips_distinct_from_missing():
    """Fitted nan category roundtrips distinct from missing."""
    # when NaN WAS seen at fit time it is an ordinary vocabulary category
    df = make_frame()  # contains NaN at fit
    rbn = ReversibleBaseN(make_encoder(df), 'biz_id')
    transport, manifest = rbn.export_frame(df)
    nan_rows = df.index[df['city'].isna()]
    assert (transport.loc[nan_rows, 'city__status'] == 'ok').all()
    result = rbn.import_frame(transport, manifest, on_violation='report')
    assert result.loc[nan_rows, 'city'].isna().all()
    assert (result.loc[nan_rows, 'city__status'] == 'ok').all()


# ---------------------------------------------------------------------------
# vocabulary version and manifest integrity
# ---------------------------------------------------------------------------


def test_wrong_vocabulary_version_is_rejected_by_fingerprint():
    """Wrong vocabulary version is rejected by fingerprint."""
    df = make_frame()
    rbn = ReversibleBaseN(make_encoder(df, base=2), 'biz_id')
    other = ReversibleBaseN(make_encoder(df, base=3), 'biz_id')
    transport, other_manifest = other.export_frame(df)
    with pytest.raises(VocabularyVersionMismatchError):
        rbn.import_frame(
            transport,
            other_manifest,
            expected_fingerprint=rbn.vocabulary_fingerprint,
        )


def test_tampered_vocabulary_fails_integrity():
    """Tampered vocabulary fails integrity."""
    df = make_frame()
    rbn = ReversibleBaseN(make_encoder(df), 'biz_id')
    transport, manifest = rbn.export_frame(df)
    manifest['vocabulary']['base'] = 7
    with pytest.raises(VocabularyIntegrityError):
        rbn.import_frame(transport, manifest)


def test_tampered_manifest_checksum_fails():
    """Tampered manifest checksum fails."""
    df = make_frame()
    rbn = ReversibleBaseN(make_encoder(df), 'biz_id')
    transport, manifest = rbn.export_frame(df)
    manifest['manifest_checksum'] = '0' * 64
    with pytest.raises(VocabularyIntegrityError):
        rbn.import_frame(transport, manifest)


def test_tampered_row_id_list_fails_integrity():
    """Tampered row id list fails integrity."""
    df = make_frame()
    rbn = ReversibleBaseN(make_encoder(df), 'biz_id')
    transport, manifest = rbn.export_frame(df)
    manifest['exported_ids'][0] = {'__nan__': True}
    with pytest.raises(VocabularyIntegrityError):
        rbn.import_frame(transport, manifest)


def test_missing_whole_rows_are_detected_when_required():
    """Missing whole rows are detected when required."""
    df = make_frame(with_missing=False)
    rbn = ReversibleBaseN(make_encoder(df), 'biz_id')
    transport, manifest = rbn.export_frame(df)
    partial = transport.iloc[5:].reset_index(drop=True)
    with pytest.raises(MissingRowsError):
        rbn.import_frame(partial, manifest, require_all_exported_rows=True)


def test_schema_mismatch_raises():
    """Schema mismatch raises."""
    df = make_frame()
    rbn = ReversibleBaseN(make_encoder(df), 'biz_id')
    transport, manifest = rbn.export_frame(df)
    with pytest.raises(SchemaMismatchError):
        rbn.import_frame(transport.drop(columns=['city_0']), manifest)
    with pytest.raises(SchemaMismatchError):
        rbn.import_frame(transport.assign(extra=1), manifest)


# ---------------------------------------------------------------------------
# construction guards
# ---------------------------------------------------------------------------


def test_requires_fitted_encoder():
    """Requires fitted encoder."""
    with pytest.raises(ValueError):
        ReversibleBaseN(BaseNEncoder(cols=['city']), 'id')


def test_requires_bijective_handle_strategies():
    """Requires bijective handle strategies."""
    df = make_frame()
    for kwargs in (
        {'handle_unknown': 'return_nan'},
        {'handle_missing': 'return_nan'},
        {'drop_invariant': True},
        {'return_df': False},
    ):
        encoder = BaseNEncoder(cols=['city', 'tier'], **kwargs).fit(df)
        with pytest.raises(ValueError):
            ReversibleBaseN(encoder, 'biz_id')


def test_duplicate_or_missing_row_ids_rejected():
    """Duplicate or missing row ids rejected."""
    df = make_frame()
    rbn = ReversibleBaseN(make_encoder(df), 'biz_id')
    dup = df.copy()
    dup.loc[1, 'biz_id'] = dup.loc[0, 'biz_id']
    with pytest.raises(ValueError):
        rbn.export_frame(dup)
    null_id = df.copy()
    null_id.loc[0, 'biz_id'] = np.nan
    with pytest.raises(ValueError):
        rbn.export_frame(null_id)


def test_rebuilt_encoder_matches_native_inverse_transform():
    """Rebuilt encoder matches native inverse transform."""
    # the recovery path must delegate to the genuine BaseN inverse transform
    df = make_frame(with_missing=False)
    encoder = make_encoder(df)
    rbn = ReversibleBaseN(encoder, 'biz_id')
    transport, manifest = rbn.export_frame(df)
    city_cols = manifest['fields']['city']['digit_cols']
    tier_cols = manifest['fields']['tier']['digit_cols']

    rebuilt = rbn._build_encoder_from_manifest(_single_field(manifest, 'city'))
    native_full = encoder.inverse_transform(encoder.transform(df))
    rebuilt_out = rebuilt.inverse_transform(transport[city_cols])['city']
    assert rebuilt_out.tolist() == native_full['city'].tolist()

    rebuilt_tier = rbn._build_encoder_from_manifest(_single_field(manifest, 'tier'))
    assert (
        rebuilt_tier.inverse_transform(transport[tier_cols])['tier'].tolist()
        == native_full['tier'].tolist()
    )


def _single_field(manifest, field):
    return {
        **manifest,
        'vocabulary': {
            **manifest['vocabulary'],
            'encoded_cols': [field],
            'fields': {field: manifest['vocabulary']['fields'][field]},
        },
        'fields': {field: manifest['fields'][field]},
    }


def test_manifest_is_json_serializable(tmp_path):
    """Manifest is json serializable."""
    df = make_frame()
    rbn = ReversibleBaseN(make_encoder(df), 'biz_id')
    _, manifest = rbn.export_frame(df)
    text = json.dumps(manifest)
    assert json.loads(text)['vocabulary_fingerprint'] == manifest['vocabulary_fingerprint']
