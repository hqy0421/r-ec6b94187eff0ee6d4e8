"""Reversible Base-N encoding for lossy transport channels.

This module wraps :class:`category_encoders.basen.BaseNEncoder` with an explicit
export/import contract for moving the encoded numeric columns over a channel
that may drop or corrupt individual cells:

* the vocabulary (category -> ordinal code -> base-N digits), the digit column
  order and the encoder configuration are fitted once, bound to the *business
  row id* and the *source field name*, and exported as a checksummed manifest;
* every group of digit columns belonging to one field of one row is protected
  by exactly two redundancy symbols (two exact integer checksums with position
  weights). As long as the redundancy itself arrives intact, the group
  tolerates one wrong value at an unknown position, or up to two missing
  values at known positions;
* missing source values, out-of-vocabulary categories and transport damage are
  three distinct, separately reported conditions. A row whose vocabulary
  version does not match or whose integrity checks fail is never silently
  decoded into some other category.

Recovery only repairs the numeric transport layer. The actual category lookup
is always delegated to the native :meth:`BaseNEncoder.inverse_transform`.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from category_encoders.basen import BaseNEncoder
from category_encoders.ordinal import OrdinalEncoder

__all__ = [
    'ReversibleBaseN',
    'ReversibleBaseNError',
    'VocabularyVersionMismatchError',
    'VocabularyIntegrityError',
    'SchemaMismatchError',
    'MissingRowsError',
    'UnrecoverableMissingError',
    'OutOfVocabularyError',
    'SourceMissingError',
    'UnrecoverableCorruptionError',
]

MANIFEST_FORMAT_VERSION = 1
STATUS_OK = 'ok'
STATUS_MISSING = 'missing'
STATUS_OOV = 'oov'
# internal report status for transport damage (never a legal export value)
STATUS_CORRUPTION = 'corruption'
_DECLARED_STATUSES = (STATUS_OK, STATUS_MISSING, STATUS_OOV)
# suffixes attached to the transport frame (never seen by the BaseN encoder)
STATUS_SUFFIX = '__status'
RAW_SUFFIX = '__raw'
P0_SUFFIX = '__p0'
P1_SUFFIX = '__p1'
ERROR_SUFFIX = '__error'


# ---------------------------------------------------------------------------
# typed errors: the failure families stay distinguishable by type
# ---------------------------------------------------------------------------


class ReversibleBaseNError(Exception):
    """Base class for every failure of the reversible Base-N transport."""


class VocabularyVersionMismatchError(ReversibleBaseNError):
    """The received manifest does not describe the locally fitted vocabulary."""


class VocabularyIntegrityError(ReversibleBaseNError):
    """The manifest itself was tampered with or failed its checksum."""


class SchemaMismatchError(ReversibleBaseNError):
    """Row ids / columns of the received data do not match the manifest."""


class MissingRowsError(ReversibleBaseNError):
    """Whole business rows present at export are absent at import."""


class UnrecoverableMissingError(ReversibleBaseNError):
    """More cells are missing than the two-redundancy code can restore."""


class OutOfVocabularyError(ReversibleBaseNError):
    """A source category was not part of the fitted vocabulary."""


class SourceMissingError(ReversibleBaseNError):
    """A source value was missing at export time."""


class UnrecoverableCorruptionError(ReversibleBaseNError):
    """Integrity checks failed and no value within the code guarantee fits."""


# ---------------------------------------------------------------------------
# serialization / number-theory helpers
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    """JSON fallback covering the scalar types a pandas index may hold."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return {'__timestamp__': obj.isoformat()}
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f'Object of type {type(obj)!r} is not JSON serializable')


def _canonical_json(obj: Any) -> str:
    """Stable JSON text used as the input of every checksum."""
    return json.dumps(
        obj, sort_keys=True, ensure_ascii=False, separators=(',', ':'), default=_json_default
    )


def _sha256(obj: Any) -> str:
    return hashlib.sha256(_canonical_json(obj).encode('utf-8')).hexdigest()


_NAN_TOKEN = {'__nan__': True}


def _tokenize_scalar(value: Any) -> Any:
    """Turn a vocabulary category into a JSON-safe, NaN-preserving token."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return dict(_NAN_TOKEN)
    if isinstance(value, np.generic):
        return _tokenize_scalar(value.item())
    if isinstance(value, (int, float, bool, str)):
        return value
    if isinstance(value, pd.Timestamp):
        return {'__timestamp__': value.isoformat()}
    raise TypeError(
        f'Vocabulary category of type {type(value)!r} cannot be serialized; '
        'use str/int/float/bool/Timestamp categories'
    )


def _detokenize_scalar(token: Any) -> Any:
    if isinstance(token, dict):
        if token == _NAN_TOKEN:
            return np.nan
        if '__timestamp__' in token:
            return pd.Timestamp(token['__timestamp__'])
    return token


def _membership_mask(series: pd.Series, vocab_index: pd.Index) -> pd.Series:
    """Boolean mask of values belonging to the fitted vocabulary.

    NaN membership is handled explicitly because ``Series.isin`` drops NaN
    matches on object indexes across pandas versions.
    """
    regular_values = [v for v in vocab_index if isinstance(v, str) or not pd.isna(v)]
    result = series.isin(regular_values)
    if pd.isna(vocab_index).any():
        result = result | series.isna()
    return result


def _resolve_dtype(name: str) -> Any:
    try:
        return np.dtype(name)
    except TypeError:
        return object


def _real_digit_patterns(fmeta: dict[str, Any]) -> set[tuple[int, ...]]:
    """Digit tuples that correspond to real (non-sentinel) vocabulary rows."""
    table = fmeta['digit_table']
    return {tuple(int(v) for v in table[code]) for code in table if int(code) >= 0}


def _single_field_manifest(manifest: dict[str, Any], field: str) -> dict[str, Any]:
    """Project a manifest onto one field for a single-field native encoder."""
    return {
        **manifest,
        'vocabulary': {
            **manifest['vocabulary'],
            'encoded_cols': [field],
            'fields': {field: manifest['vocabulary']['fields'][field]},
        },
        'fields': {field: manifest['fields'][field]},
    }


# ---------------------------------------------------------------------------
# main class
# ---------------------------------------------------------------------------


class ReversibleBaseN:
    """Reversible, integrity-protected export/import around a BaseN encoder.

    Parameters
    ----------
    encoder: BaseNEncoder
        A fitted ``BaseNEncoder``. Only the bijective configuration is
        supported: ``handle_unknown='value'``, ``handle_missing='value'``,
        ``drop_invariant=False`` and the default ``return_df=True``.
    id_col: str
        Name of the business row-id column. It is never encoded and binds
        every exported/imported group to its source row.

    Notes
    -----
    The two redundancy symbols per field/row group are exact integer
    checksums (no modular reduction), with position weights 1..k::

        p0 = sum(d_i)
        p1 = sum((i + 1) * d_i)

    Because every digit is a non-negative integer below ``base`` and the
    checksums are stored without wrapping, the two underlying linear equations
    have a unique integer solution. With the checksums intact the group
    therefore corrects one wrong digit at an unknown position (the syndrome
    ratio ``s1 / s0`` gives the position weight) and up to two missing digits
    at known positions. Every recovered value is re-checked against the
    checksums and the fitted vocabulary, so a damaged group is never silently
    decoded into another category.
    """

    def __init__(self, encoder: BaseNEncoder, id_col: str):
        if not isinstance(encoder, BaseNEncoder):
            raise TypeError('encoder must be a fitted category_encoders.BaseNEncoder')
        if getattr(encoder, 'mapping', None) is None:
            raise ValueError('encoder must be fitted before use')
        if encoder.handle_unknown != 'value' or encoder.handle_missing != 'value':
            raise ValueError(
                "Reversible transport requires handle_unknown='value' and "
                "handle_missing='value' so missing and unseen categories stay distinguishable"
            )
        if encoder.drop_invariant:
            raise ValueError('Reversible transport requires drop_invariant=False')
        if not encoder.return_df:
            raise ValueError('Reversible transport requires return_df=True')
        if not isinstance(id_col, str) or not id_col:
            raise ValueError('id_col must be a non-empty column name')
        if id_col in encoder.cols:
            raise ValueError(f'id_col {id_col!r} must not be one of the encoded columns')

        self.encoder = encoder
        self.id_col = id_col
        self.base = int(encoder.base)
        self._field_specs = self._build_field_specs(encoder)
        self._vocab_doc = self._build_vocabulary_doc(encoder)
        self.vocabulary_fingerprint = _sha256(self._vocab_doc)

    # ------------------------------------------------------------------
    # introspection of the fitted native encoder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_field_specs(encoder: BaseNEncoder) -> dict[str, dict[str, Any]]:
        specs: dict[str, dict[str, Any]] = {}
        for switch in encoder.mapping:
            field = switch['col']
            mapping_df = switch['mapping']
            digit_cols = list(mapping_df.columns)
            real_codes = sorted(int(code) for code in mapping_df.index if int(code) >= 0)
            if not real_codes:
                raise ValueError(
                    f'field {field!r} has no fitted categories; nothing reversible to export'
                )
            specs[field] = {
                'field': field,
                'digit_cols': digit_cols,
                'n_digits': len(digit_cols),
                'real_codes': real_codes,
                'max_ordinal': max(real_codes),
            }
        return specs

    def _build_vocabulary_doc(self, encoder: BaseNEncoder) -> dict[str, Any]:
        """Full JSON-safe description required to rebuild the native encoder."""
        fields: dict[str, Any] = {}
        for ord_switch, basen_switch in zip(
            encoder.ordinal_encoder.mapping, encoder.mapping, strict=True
        ):
            field = basen_switch['col']
            ord_series = ord_switch['mapping']
            mapping_df = basen_switch['mapping']

            categories = [
                {
                    'token': _tokenize_scalar(index_value),
                    'ordinal_code': int(code),
                }
                for index_value, code in ord_series.items()
            ]
            digit_table = {
                str(int(code)): [int(v) for v in row] for code, row in mapping_df.iterrows()
            }
            fields[field] = {
                'data_type': str(ord_switch['data_type']),
                'index_dtype': str(ord_series.index.dtype),
                'categories': categories,
                'digit_cols': list(mapping_df.columns),
                'digit_table': digit_table,
            }
        return {
            'base': self.base,
            'handle_unknown': encoder.handle_unknown,
            'handle_missing': encoder.handle_missing,
            'encoded_cols': list(encoder.cols),
            'fields': fields,
        }

    def _build_manifest(self, id_series: pd.Series) -> dict[str, Any]:
        field_manifests = {}
        for field, spec in self._field_specs.items():
            k = spec['n_digits']
            field_manifests[field] = {
                'field': field,
                'digit_cols': spec['digit_cols'],
                'n_digits': k,
                'base': self.base,
                'p0_col': field + P0_SUFFIX,
                'p1_col': field + P1_SUFFIX,
                'status_col': field + STATUS_SUFFIX,
                'raw_col': field + RAW_SUFFIX,
                'max_ordinal': spec['max_ordinal'],
                'digit_table': self._vocab_doc['fields'][field]['digit_table'],
                # exact integer bounds, used by the receiver to reject a
                # checksum that could never have been produced at export time
                'max_p0': k * (self.base - 1) if self.base > 1 else 1,
                'max_p1': ((self.base - 1) * k * (k + 1) // 2 if self.base > 1 else k),
            }
        exported_ids = [_tokenize_scalar(value) for value in id_series.tolist()]
        body = {
            'format_version': MANIFEST_FORMAT_VERSION,
            'id_col': self.id_col,
            'id_dtype': str(id_series.dtype),
            'exported_ids': exported_ids,
            'exported_ids_checksum': _sha256(exported_ids),
            'vocabulary': self._vocab_doc,
            'fields': field_manifests,
        }
        body['vocabulary_fingerprint'] = _sha256(body['vocabulary'])
        body['manifest_checksum'] = _sha256(
            {k: v for k, v in body.items() if k != 'manifest_checksum'}
        )
        return body

    # ------------------------------------------------------------------
    # checksum arithmetic
    # ------------------------------------------------------------------

    @staticmethod
    def _checksums(digits: list[int]) -> tuple[int, int]:
        # exact integer checksums; deliberately no modular reduction so the
        # recovery equations stay ordinary integer linear equations
        p0 = sum(int(d) for d in digits)
        p1 = sum((i + 1) * int(d) for i, d in enumerate(digits))
        return p0, p1

    @staticmethod
    def _digit_in_alphabet(digit: int, base: int) -> bool:
        # base 1 is the one-hot convention: every "digit" is 0 or 1
        if base == 1:
            return digit in (0, 1)
        return 0 <= digit < base

    @classmethod
    def _validate_alphabet(cls, digits: list[int], base: int) -> bool:
        return all(cls._digit_in_alphabet(int(d), base) for d in digits)

    # ------------------------------------------------------------------
    # export
    # ------------------------------------------------------------------

    def export_frame(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Encode ``df`` and return ``(transport_frame, manifest)``.

        Missing source values and out-of-vocabulary categories are not encoded
        at all: their digit cells stay empty and an explicit per-field status
        column records which of the two conditions applies.
        """
        if self.id_col not in df.columns:
            raise KeyError(f'id column {self.id_col!r} is missing from the input frame')
        if df[self.id_col].isna().any():
            raise ValueError('business row ids must not be missing')
        if df[self.id_col].duplicated().any():
            dupes = df.loc[df[self.id_col].duplicated(), self.id_col].tolist()
            raise ValueError(f'business row ids must be unique; duplicates: {dupes!r}')
        for field in self._field_specs:
            if field not in df.columns:
                raise KeyError(f'encoded field {field!r} is missing from the input frame')

        work = df.reset_index(drop=True)
        manifest = self._build_manifest(work[self.id_col])

        # encode the whole frame once; only ok rows will use these digits
        encoded = self.encoder.transform(work)
        vocab_indexes = {
            field: switch['mapping'].index
            for field, switch in zip(
                self._field_specs, self.encoder.ordinal_encoder.mapping, strict=True
            )
        }

        # export-side membership per (field, row) for the status columns
        known_masks = {
            field: _membership_mask(work[field], vocab_indexes[field])
            for field in self._field_specs
        }

        # NaN is only a *real* category when it was assigned a positive
        # ordinal code at fit time. A NaN -> MISSING_SENTINEL (-2) row is the
        # placeholder the ordinal encoder always keeps, even if NaN never
        # appeared in the fit data; such a NaN at export is a true missing.
        ordinal_series = {
            switch['col']: switch['mapping'] for switch in self.encoder.ordinal_encoder.mapping
        }
        nan_is_category = {}
        for field in self._field_specs:
            series = ordinal_series[field]
            nan_codes = series[series.index.isna()]
            nan_is_category[field] = bool(len(nan_codes) and int(nan_codes.iloc[0]) >= 0)

        records: dict[str, list[Any]] = {self.id_col: work[self.id_col].tolist()}
        ordered_cols = [self.id_col]

        for field, spec in self._field_specs.items():
            fmeta = manifest['fields'][field]
            digit_cols = spec['digit_cols']
            ordered_cols += digit_cols + [
                fmeta['p0_col'],
                fmeta['p1_col'],
                fmeta['status_col'],
                fmeta['raw_col'],
            ]
            for col in digit_cols:
                records[col] = []
            records[fmeta['p0_col']] = []
            records[fmeta['p1_col']] = []
            records[fmeta['status_col']] = []
            records[fmeta['raw_col']] = []

        for pos, (_, row) in enumerate(work.iterrows()):
            for field, spec in self._field_specs.items():
                fmeta = manifest['fields'][field]
                digit_cols = spec['digit_cols']
                value = row[field]
                is_missing = pd.isna(value)
                is_known = bool(known_masks[field].iloc[pos])

                # A NaN that was part of the fitted vocabulary is a real
                # category with a positive ordinal code; only a NaN outside
                # the vocabulary is a genuine source-level missing value.
                if is_missing and not (nan_is_category[field] and is_known):
                    status, raw_value = STATUS_MISSING, np.nan
                elif not is_known:
                    status, raw_value = STATUS_OOV, value
                else:
                    status, raw_value = STATUS_OK, np.nan

                records[fmeta['status_col']].append(status)
                records[fmeta['raw_col']].append(raw_value)

                if status == STATUS_OK:
                    digits = [int(encoded.iloc[pos][c]) for c in digit_cols]
                    p0, p1 = self._checksums(digits)
                    for col, digit in zip(digit_cols, digits, strict=True):
                        records[col].append(digit)
                    records[fmeta['p0_col']].append(p0)
                    records[fmeta['p1_col']].append(p1)
                else:
                    # no redundancy is emitted for rows that are never decoded
                    for col in digit_cols:
                        records[col].append(np.nan)
                    records[fmeta['p0_col']].append(np.nan)
                    records[fmeta['p1_col']].append(np.nan)

        transport = pd.DataFrame(records, columns=ordered_cols)
        return transport, manifest

    def to_files(
        self,
        df: pd.DataFrame,
        data_path: str | os.PathLike,
        manifest_path: str | os.PathLike,
    ) -> dict[str, Any]:
        """Persist a CSV transport frame and a JSON manifest."""
        transport, manifest = self.export_frame(df)
        data_path, manifest_path = Path(data_path), Path(manifest_path)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        transport.to_csv(data_path, index=False)
        with open(manifest_path, 'w', encoding='utf-8') as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, default=_json_default)
        return manifest

    # ------------------------------------------------------------------
    # import
    # ------------------------------------------------------------------

    def import_frame(
        self,
        transport: pd.DataFrame,
        manifest: dict[str, Any],
        on_violation: str = 'raise',
        expected_fingerprint: str | None = None,
        require_all_exported_rows: bool = False,
    ) -> pd.DataFrame:
        """Verify, repair and decode a transport frame.

        Parameters
        ----------
        transport: DataFrame
            Numeric frame produced by :meth:`export_frame`, possibly with
            missing or corrupted cells.
        manifest: dict
            Manifest produced together with the frame.
        on_violation: {'raise', 'report'}
            ``'raise'`` (default) raises the first typed row-level error.
            ``'report'`` never raises for row-level conditions: problematic
            rows keep ``NaN`` category values and are described in per-field
            ``__status`` / ``__error`` columns. Manifest-level failures always
            raise.
        expected_fingerprint: str, optional
            Fingerprint of the vocabulary the caller expects. A mismatch
            raises :class:`VocabularyVersionMismatchError`, guarding against
            decoding with the wrong vocabulary version.
        require_all_exported_rows: bool, default False
            When True, raise :class:`MissingRowsError` if the frame does not
            cover every row id listed in the manifest.
        """
        if on_violation not in ('raise', 'report'):
            raise ValueError("on_violation must be 'raise' or 'report'")
        self._verify_manifest(manifest, expected_fingerprint)
        frame = transport.copy(deep=True).reset_index(drop=True)
        self._check_schema(frame, manifest)

        id_col = manifest['id_col']
        if frame[id_col].isna().any():
            raise SchemaMismatchError('business row ids must not be missing at import')
        if frame[id_col].duplicated().any():
            raise SchemaMismatchError('business row ids must be unique at import')

        out = pd.DataFrame({id_col: frame[id_col]})
        # category columns start empty (NaN) and are filled with native-decoded
        # values, so keep them object-typed to avoid mixed-dtype assignments
        for field in manifest['fields']:
            out[field] = pd.Series([np.nan] * len(frame), dtype=object)
        # one repaired row per position: digit vectors for every field, or
        # None when the group could not be recovered
        repaired: dict[int, dict[str, list[int]]] = {}
        reports: dict[str, list[Any]] = {}
        for field in manifest['fields']:
            reports[field + STATUS_SUFFIX] = []
            reports[field + ERROR_SUFFIX] = []

        for pos, (_, row) in enumerate(frame.iterrows()):
            row_id = row[id_col]
            for field, fmeta in manifest['fields'].items():
                status, error, digits = self._recover_group(row, field, fmeta)
                reports[field + STATUS_SUFFIX].append(status)
                reports[field + ERROR_SUFFIX].append(error)
                if digits is not None:
                    repaired.setdefault(pos, {})[field] = digits
                if on_violation == 'raise' and status != STATUS_OK:
                    raise _error_for(status, field, row_id, error)

        if require_all_exported_rows:
            self._check_row_coverage(frame, manifest)

        # decode each field independently through the NATIVE inverse_transform,
        # so an unrecoverable group in one field never withholds the correctly
        # recovered categories of the other fields. Each field gets its own
        # single-field native encoder rebuilt from the same manifest.
        for field, fmeta in manifest['fields'].items():
            field_manifest = _single_field_manifest(manifest, field)
            field_encoder = self._build_encoder_from_manifest(field_manifest)
            ok_positions = [pos for pos in range(len(frame)) if field in repaired.get(pos, {})]
            if not ok_positions:
                continue
            block = pd.DataFrame(
                [repaired[pos][field] for pos in ok_positions],
                columns=fmeta['digit_cols'],
                index=frame.loc[ok_positions, id_col].to_numpy(),
            )
            decoded = field_encoder.inverse_transform(block)
            self._assert_decode_in_vocabulary(decoded[field], field, manifest)
            for offset, pos in enumerate(ok_positions):
                out.iat[pos, out.columns.get_loc(field)] = decoded[field].iloc[offset]

        if on_violation == 'report':
            for col, values in reports.items():
                out[col] = values
        return out

    def from_files(
        self,
        data_path: str | os.PathLike,
        manifest_path: str | os.PathLike,
        on_violation: str = 'raise',
        expected_fingerprint: str | None = None,
        require_all_exported_rows: bool = False,
    ) -> pd.DataFrame:
        """Load a CSV/JSON pair and run :meth:`import_frame`."""
        with open(manifest_path, encoding='utf-8') as handle:
            manifest = json.load(handle)
        transport = pd.read_csv(data_path)
        return self.import_frame(
            transport,
            manifest,
            on_violation=on_violation,
            expected_fingerprint=expected_fingerprint,
            require_all_exported_rows=require_all_exported_rows,
        )

    # ------------------------------------------------------------------
    # manifest verification and native encoder reconstruction
    # ------------------------------------------------------------------

    @staticmethod
    def _verify_manifest(manifest: dict[str, Any], expected_fingerprint: str | None) -> None:
        required = {
            'format_version',
            'id_col',
            'id_dtype',
            'exported_ids',
            'exported_ids_checksum',
            'vocabulary',
            'fields',
            'vocabulary_fingerprint',
            'manifest_checksum',
        }
        missing = required - set(manifest)
        if missing:
            raise VocabularyIntegrityError(f'manifest is missing keys: {sorted(missing)!r}')
        if manifest['format_version'] != MANIFEST_FORMAT_VERSION:
            raise VocabularyVersionMismatchError(
                f'unsupported format version {manifest["format_version"]!r}; '
                f'expected {MANIFEST_FORMAT_VERSION}'
            )
        actual_fp = _sha256(manifest['vocabulary'])
        if actual_fp != manifest['vocabulary_fingerprint']:
            raise VocabularyIntegrityError(
                'vocabulary fingerprint mismatch: the manifest vocabulary was modified'
            )
        if _sha256(manifest['exported_ids']) != manifest['exported_ids_checksum']:
            raise VocabularyIntegrityError(
                'exported row-id list failed its checksum; the manifest is corrupted'
            )
        if expected_fingerprint is not None and actual_fp != expected_fingerprint:
            raise VocabularyVersionMismatchError(
                'the received vocabulary does not match the expected vocabulary version'
            )
        checksum = _sha256({k: v for k, v in manifest.items() if k != 'manifest_checksum'})
        if checksum != manifest['manifest_checksum']:
            raise VocabularyIntegrityError(
                'manifest checksum mismatch: the manifest is corrupted or tampered with'
            )

    def _build_encoder_from_manifest(self, manifest: dict[str, Any]) -> BaseNEncoder:
        """Rebuild a fitted native BaseNEncoder purely from the manifest."""
        vocab = manifest['vocabulary']
        base = int(vocab['base'])
        cols = list(vocab['encoded_cols'])

        ordinal_mapping: list[dict[str, Any]] = []
        basen_mapping: list[dict[str, Any]] = []
        for field in cols:
            fdoc = vocab['fields'][field]
            categories = [
                (_detokenize_scalar(item['token']), int(item['ordinal_code']))
                for item in fdoc['categories']
            ]
            index = pd.Index(
                [cat for cat, _ in categories],
                dtype=_resolve_dtype(fdoc['index_dtype']),
            )
            codes = pd.Series([code for _, code in categories], index=index)
            ordinal_mapping.append(
                {
                    'col': field,
                    'mapping': codes,
                    'data_type': _resolve_dtype(fdoc['data_type']),
                }
            )

            digit_cols = fdoc['digit_cols']
            rows = {int(code): fdoc['digit_table'][str(code)] for code in fdoc['digit_table']}
            mapping_df = pd.DataFrame.from_dict(
                rows, orient='index', columns=digit_cols
            ).sort_index()
            mapping_df.index.name = None
            basen_mapping.append({'col': field, 'mapping': mapping_df})

        ordinal = OrdinalEncoder(
            cols=cols,
            handle_unknown='value',
            handle_missing='value',
            mapping=ordinal_mapping,
        )
        out_names: list[str] = []
        for field in cols:
            out_names.extend(vocab['fields'][field]['digit_cols'])
        digit_width = len(out_names)

        # The ordinal stage receives the digit block; the BaseN stage first
        # collapses each field's digit columns into one ordinal column via
        # basen_to_integer, so its _dim is the number of fields.
        ordinal.mapping_supplied = True
        ordinal._dim = len(cols)
        ordinal.feature_names_in_ = list(out_names)
        ordinal.n_features_in_ = digit_width
        ordinal.feature_names_out_ = np.array(cols)
        ordinal.feature_dtypes_in_ = [m['data_type'] for m in ordinal_mapping]
        ordinal.cols = list(cols)

        encoder = BaseNEncoder(
            cols=cols,
            base=base,
            handle_unknown='value',
            handle_missing='value',
            return_df=True,
            drop_invariant=False,
        )
        encoder.ordinal_encoder = ordinal
        encoder.mapping = basen_mapping
        encoder.cols = list(cols)
        encoder._dim = len(cols)
        encoder.feature_names_in_ = list(out_names)
        encoder.n_features_in_ = digit_width
        encoder.feature_dtypes_in_ = [m['data_type'] for m in ordinal_mapping]
        encoder.feature_names_out_ = np.array(out_names)
        encoder.invariant_cols = []
        return encoder

    # ------------------------------------------------------------------
    # schema / coverage checks
    # ------------------------------------------------------------------

    @staticmethod
    def _check_schema(frame: pd.DataFrame, manifest: dict[str, Any]) -> None:
        id_col = manifest['id_col']
        if id_col not in frame.columns:
            raise SchemaMismatchError(f'id column {id_col!r} is missing from the received frame')
        expected: list[str] = [id_col]
        for fmeta in manifest['fields'].values():
            expected += fmeta['digit_cols']
            expected += [
                fmeta['p0_col'],
                fmeta['p1_col'],
                fmeta['status_col'],
                fmeta['raw_col'],
            ]
        absent = [c for c in expected if c not in frame.columns]
        if absent:
            raise SchemaMismatchError(f'received frame is missing columns: {absent!r}')
        unexpected = [c for c in frame.columns if c not in expected]
        if unexpected:
            raise SchemaMismatchError(f'received frame has unknown columns: {unexpected!r}')

    @staticmethod
    def _check_row_coverage(frame: pd.DataFrame, manifest: dict[str, Any]) -> None:
        expected = {_canonical_json(token) for token in manifest['exported_ids']}
        received = {
            _canonical_json(_tokenize_scalar(value)) for value in frame[manifest['id_col']].tolist()
        }
        absent = sorted(expected - received)
        if absent:
            raise MissingRowsError(
                f'{len(absent)} exported business row(s) are absent from the received frame; '
                f'first missing ids: {absent[:5]}'
            )

    # ------------------------------------------------------------------
    # per-group recovery: the [k+2, k] decoder
    # ------------------------------------------------------------------

    def _recover_group(
        self, row: pd.Series, field: str, fmeta: dict[str, Any]
    ) -> tuple[str, str | None, list[int] | None]:
        """Return ``(status, error, recovered_digits)`` for one field/row group."""
        digit_cols: list[str] = fmeta['digit_cols']
        n_digits = int(fmeta['n_digits'])
        base = int(fmeta['base'])
        p0_col, p1_col = fmeta['p0_col'], fmeta['p1_col']
        status_col = fmeta['status_col']

        declared = row[status_col]
        if declared not in _DECLARED_STATUSES:
            return (
                STATUS_CORRUPTION,
                f'status marker {declared!r} is not one of {_DECLARED_STATUSES}',
                None,
            )

        # non-ok rows carry no redundancy and must never be decoded
        if declared == STATUS_MISSING:
            return STATUS_MISSING, 'source value was missing at export', None
        if declared == STATUS_OOV:
            return STATUS_OOV, 'source category was outside the fitted vocabulary', None

        p0, p1 = row[p0_col], row[p1_col]
        if pd.isna(p0) or pd.isna(p1):
            return (
                STATUS_CORRUPTION,
                'redundancy symbols are missing, so the group cannot be verified',
                None,
            )
        p0, p1 = int(p0), int(p1)
        # exact integer checksums have fixed, manifest-known ranges
        if not (0 <= p0 <= int(fmeta['max_p0'])):
            return (
                STATUS_CORRUPTION,
                f'p0 checksum {p0} is outside the export range [0, {fmeta["max_p0"]}]',
                None,
            )
        if not (0 <= p1 <= int(fmeta['max_p1'])):
            return (
                STATUS_CORRUPTION,
                f'p1 checksum {p1} is outside the export range [0, {fmeta["max_p1"]}]',
                None,
            )

        raw_digits = [row[c] for c in digit_cols]
        missing_positions = [i for i, d in enumerate(raw_digits) if pd.isna(d)]
        present = {i: int(raw_digits[i]) for i in range(n_digits) if i not in missing_positions}
        for i, digit in present.items():
            if not self._digit_in_alphabet(digit, base):
                return (
                    STATUS_CORRUPTION,
                    f'digit at position {i} is outside the base-{base} alphabet',
                    None,
                )

        if missing_positions:
            # erasure mode: missing cells at known positions (max two)
            if len(missing_positions) > 2:
                return (
                    STATUS_MISSING,
                    f'{len(missing_positions)} cells missing; the code restores at most two',
                    None,
                )
            digits = self._restore_erasures(present, missing_positions, p0, p1, n_digits)
            if digits is None:
                return (
                    STATUS_CORRUPTION,
                    'checksums are inconsistent with any value at the missing positions',
                    None,
                )
        else:
            # error mode: every cell present, at most one may be wrong
            digits = [present[i] for i in range(n_digits)]
            s0 = p0 - sum(digits)
            s1 = p1 - sum((i + 1) * d for i, d in enumerate(digits))
            if s0 != 0 or s1 != 0:
                position = self._locate_error(s1, s0, n_digits)
                if position is None:
                    return (
                        STATUS_CORRUPTION,
                        'syndromes do not identify a single digit position; damage exceeds '
                        'the one-error correction guarantee',
                        None,
                    )
                candidate = digits[position] + s0
                if not self._digit_in_alphabet(candidate, base):
                    return (
                        STATUS_CORRUPTION,
                        f'the only single-error correction yields an invalid base-{base} digit',
                        None,
                    )
                digits[position] = candidate
                check0, check1 = self._checksums(digits)
                if check0 != p0 or check1 != p1:
                    return (
                        STATUS_CORRUPTION,
                        'correction failed re-verification',
                        None,
                    )

        if not self._validate_alphabet(digits, base):
            return (
                STATUS_CORRUPTION,
                f'restored digits contain a value outside the base-{base} alphabet',
                None,
            )
        # the digit table covers every ordinal row including the all-zero
        # unknown/missing sentinels; a recovered group is only trustworthy when
        # its digits match a row with a non-negative code
        if tuple(digits) not in _real_digit_patterns(fmeta):
            return (
                STATUS_CORRUPTION,
                'recovered digits do not match any fitted vocabulary category',
                None,
            )
        return STATUS_OK, None, digits

    @staticmethod
    def _locate_error(s1: int, s0: int, n_digits: int) -> int | None:
        """Locate the single error from the exact integer syndrome ratio.

        For one flipped digit at position ``j`` with difference ``delta``:
        ``s0 = delta`` and ``s1 = (j + 1) * delta``, so the position weight is
        ``s1 / s0``. A non-integer or out-of-range ratio means the damage is
        not a correctable single-symbol error.
        """
        if s0 == 0:
            return None
        if s1 % s0 != 0:
            return None
        weight = s1 // s0
        if not 1 <= weight <= n_digits:
            return None
        return weight - 1

    @staticmethod
    def _restore_erasures(
        present: dict[int, int],
        missing_positions: list[int],
        p0: int,
        p1: int,
        n_digits: int,
    ) -> list[int] | None:
        """Solve the two exact integer checksum equations for 1 or 2 erasures."""
        digits = [0] * n_digits
        for i, digit in present.items():
            digits[i] = digit
        rhs0 = p0 - sum(present.values())
        rhs1 = p1 - sum((i + 1) * d for i, d in present.items())

        if len(missing_positions) == 1:
            pos = missing_positions[0]
            weight = pos + 1
            value = rhs0
            if weight * value != rhs1:
                return None
            digits[pos] = value
            return digits

        pos_a, pos_b = missing_positions  # always ordered a < b
        weight_a, weight_b = pos_a + 1, pos_b + 1
        determinant = weight_b - weight_a  # positive, never zero
        numerator = weight_b * rhs0 - rhs1
        if numerator % determinant != 0:
            return None
        value_a = numerator // determinant
        value_b = rhs0 - value_a
        if weight_a * value_a + weight_b * value_b != rhs1:
            return None
        digits[pos_a], digits[pos_b] = value_a, value_b
        return digits

    @staticmethod
    def _assert_decode_in_vocabulary(
        decoded: pd.Series, field: str, manifest: dict[str, Any]
    ) -> None:
        # Only non-NaN decoded values need checking here. A NaN result is
        # legitimate when NaN itself is a fitted category (the native inverse
        # transform maps its positive ordinal code back to NaN because pandas
        # cannot use NaN as a map target). A genuine failure would already have
        # been rejected earlier in _recover_group via the real-digit-pattern
        # check, which excludes the all-zero unknown/missing sentinel rows.
        nan_is_category = any(
            item['token'] == _NAN_TOKEN and item['ordinal_code'] >= 0
            for item in manifest['vocabulary']['fields'][field]['categories']
        )
        valid_tokens = {
            _canonical_json(item['token'])
            for item in manifest['vocabulary']['fields'][field]['categories']
            if item['ordinal_code'] >= 0
        }
        for value in decoded:
            if pd.isna(value):
                if not nan_is_category:
                    raise UnrecoverableCorruptionError(
                        f'{field}: native inverse_transform produced NaN although NaN is not '
                        'a fitted category; refusing to silently decode it'
                    )
                continue
            if _canonical_json(_tokenize_scalar(value)) not in valid_tokens:
                raise UnrecoverableCorruptionError(
                    f'{field}: decoded value {value!r} is not a fitted vocabulary category'
                )


def _error_for(status: str, field: str, row_id: Any, detail: str | None) -> ReversibleBaseNError:
    message = f'field {field!r}, row id {row_id!r}: {detail}'
    if status == STATUS_MISSING:
        if detail and 'at most two' in detail:
            return UnrecoverableMissingError(message)
        return SourceMissingError(message)
    if status == STATUS_OOV:
        return OutOfVocabularyError(message)
    return UnrecoverableCorruptionError(message)
