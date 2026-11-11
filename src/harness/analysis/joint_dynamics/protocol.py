from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Literal, Mapping, NewType

WORKSPACE = Path('/home/aditya/experiments/world-model-r38-20261026')
REPO = WORKSPACE / 'repo'
PLANNING = REPO / 'outputs/r38-joint-dynamics/planning'
CHECKPOINT_SHA256 = '65a4c4db2ca1b44ebf29fa12659fbd09dea6ede606ce1d3279535144933a001a'
ROW_SHA256 = '07822fed97840e5c2ccec048d8b26e981bb40b3ff001ad647eb2850c9e61ef9f'
COMPARTMENTS = ('necrotic_non_enhancing', 'enhancing', 'edema_flair')
CONDITIONS = ('identity', 'intensity', 'resolution')
CALIBRATION_GRID = (0.5, 1.0, 2.0)
OBSERVATION_SD = 0.05


@dataclass(frozen=True)
class Arm:
    name: str
    joint: bool
    history: bool


ARMS = (Arm('OH', False, True), Arm('OL', False, False),
        Arm('JH', True, True), Arm('JL', True, False))


@dataclass(frozen=True)
class Plan:
    arms: tuple[Arm, ...] = ARMS
    seeds: tuple[int, ...] = (42, 43, 44)
    epochs: int = 20
    updates_per_epoch: int = 262
    batch_size: int = 1
    history_slots: int = 4
    widths: tuple[int, ...] = (16, 32, 64, 128)
    hidden_width: int = 64
    input_width: int = 247
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    clipping_norm: float = 1.0
    future_mask_weight: float = 0.1
    observation_sd: float = OBSERVATION_SD
    calibration_grid: tuple[float, ...] = CALIBRATION_GRID
    normalizer_ddof: int = 0
    normalizer_floor: float = 0.5
    source_visits: int = 353
    cohort_counts: tuple[tuple[str, int, int], ...] = (
        ('source52', 262, 52), ('validation13', 60, 13),
        ('primary', 75, 18), ('single', 26, 23), ('nuisance', 18, 18))
    reconstruction_tolerance: float = 1e-4
    normalizer_tolerance: float = 1e-6
    precision: str = 'BF16-autocast/FP32-loss/no-scaler/no-fallback'
    deterministic: str = 'warn_only;CUBLAS=:4096:8;TF32=off;cudnn-deterministic'
    bootstrap_draws: int = 10000
    bootstrap_seed: int = 2026
    adjusted_confidence: float = 0.995
    companion_confidence: float = 0.95
    relative_gain: float = 0.05
    retention_mae_margin: float = 0.05
    retention_dice_margin: float = 0.02
    coverage80_bounds: tuple[float, float] = (0.7, 0.9)
    nuisance_crps_margin: float = 0.1
    nuisance_mae_margin: float = 0.05

    @property
    def total_updates(self) -> int:
        return len(self.arms) * len(self.seeds) * self.epochs * self.updates_per_epoch


PLAN = Plan()


@dataclass(frozen=True)
class Normalizer:
    visit_ids: tuple[str, ...]
    patient_ids: tuple[str, ...]
    mean: tuple[float, float, float]
    scale: tuple[float, float, float]


@dataclass(frozen=True)
class ForecastRow:
    row_id: str
    patient_id: str
    cutoff_visit_id: str
    target_visit_id: str
    history_visit_ids: tuple[str, ...]
    gap_log1p: tuple[float, ...]
    history_mask: tuple[bool, ...]
    truth: tuple[float, float, float]
    cutoff_measured: tuple[float, float, float]
    primary_history: bool

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ForecastRow:
        return cls(**{key: tuple(val) if isinstance(val, list) else val
                      for key, val in value.items()})


Digest = NewType('Digest', str)
RepoPath = NewType('RepoPath', str)
PhaseName = NewType('PhaseName', str)


def _bound_digest(value: object) -> Digest:
    if not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise ValueError('invalid SHA256 digest')
    return Digest(value)


@dataclass(frozen=True)
class SourceIdentity:
    """Complete R38 implementation source, distinct from control-input data hashes."""
    files: Mapping[RepoPath, Digest]
    digest: Digest

    def __post_init__(self) -> None:
        if not isinstance(self.files, Mapping) or not self.files:
            raise ValueError('implementation source inventory is empty')
        normalized = {}
        for name, sha in self.files.items():
            if (not isinstance(name, str) or Path(name).is_absolute()
                    or '..' in Path(name).parts or Path(name).as_posix() != name):
                raise ValueError('unsafe implementation source path')
            normalized[RepoPath(name)] = _bound_digest(sha)
        object.__setattr__(self, 'files', MappingProxyType(dict(sorted(normalized.items()))))
        if _bound_digest(self.digest) != canonical_hash(dict(self.files)):
            raise ValueError('implementation source manifest/digest mismatch')

    @classmethod
    def capture(cls, repo: Path = REPO) -> SourceIdentity:
        files = source_hashes(repo)
        return cls(files, Digest(canonical_hash(files)))

    @classmethod
    def from_dict(cls, value: Mapping[str, object], *, repo: Path = REPO) -> SourceIdentity:
        if (not isinstance(value, Mapping) or set(value) != {'schema', 'files', 'digest'}
                or value['schema'] != 'r38-source-identity-v4'):
            raise ValueError('invalid implementation source identity schema')
        result = cls(value['files'], value['digest'])
        result.require_current(repo)
        return result

    def require_current(self, repo: Path = REPO) -> None:
        if self != type(self).capture(repo):
            raise ValueError('implementation source inventory differs from current code')

    def to_dict(self) -> dict[str, object]:
        return {'schema': 'r38-source-identity-v4', 'files': dict(self.files), 'digest': self.digest}


@dataclass(frozen=True)
class ArtifactRef:
    path: Path
    sha256: Digest

    def __post_init__(self) -> None:
        object.__setattr__(self, 'path', Path(self.path))
        if not self.path.is_absolute():
            raise ValueError('artifact references require absolute paths')
        _bound_digest(self.sha256)

    def verify(self) -> None:
        if not self.path.is_file() or digest(self.path) != self.sha256:
            raise ValueError('referenced artifact bytes changed')

    def to_dict(self) -> dict[str, str]:
        return {'path': str(self.path), 'sha256': self.sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ArtifactRef:
        if not isinstance(value, Mapping) or set(value) != {'path', 'sha256'}:
            raise ValueError('invalid artifact reference')
        return cls(Path(value['path']), _bound_digest(value['sha256']))


@dataclass(frozen=True)
class Cost:
    cpu_seconds: float
    wall_seconds: float
    gpu_seconds: float

    def __post_init__(self) -> None:
        for name in ('cpu_seconds', 'wall_seconds', 'gpu_seconds'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError('resource costs must be finite nonnegative seconds')
            object.__setattr__(self, name, float(value))

    def to_dict(self) -> dict[str, float]:
        return {'cpu_seconds': self.cpu_seconds, 'wall_seconds': self.wall_seconds,
                'gpu_seconds': self.gpu_seconds}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Cost:
        if not isinstance(value, Mapping) or set(value) != {'cpu_seconds', 'wall_seconds', 'gpu_seconds'}:
            raise ValueError('invalid resource cost')
        return cls(value['cpu_seconds'], value['wall_seconds'], value['gpu_seconds'])


@dataclass(frozen=True)
class StageCost:
    cost: Cost
    descendants_alive: bool


@dataclass(frozen=True)
class StageLimits:
    gpu: bool
    scientific: bool
    phase_caps: Mapping[PhaseName, Cost]
    future: Cost
    cleanup: Cost
    deadline: datetime

    def __post_init__(self) -> None:
        if type(self.gpu) is not bool or type(self.scientific) is not bool:
            raise ValueError('stage flags must be explicit booleans')
        if not isinstance(self.phase_caps, Mapping) or not {'startup', 'teardown'} <= self.phase_caps.keys():
            raise ValueError('stage lifecycle requires startup and teardown caps')
        for name, cost in self.phase_caps.items():
            if (not isinstance(name, str) or re.fullmatch(r'[a-z][a-z0-9_]{0,63}', name) is None
                    or not isinstance(cost, Cost) or cost.wall_seconds <= 0
                    or (not self.gpu and cost.gpu_seconds != 0)):
                raise ValueError('invalid phase cap or stage GPU accounting')
        if (not isinstance(self.future, Cost) or not isinstance(self.cleanup, Cost)
                or not isinstance(self.deadline, datetime) or self.deadline.tzinfo is None):
            raise ValueError('invalid future reserve, cleanup or deadline')
        object.__setattr__(self, 'phase_caps', MappingProxyType(dict(self.phase_caps)))

    def to_dict(self) -> dict[str, object]:
        return {'gpu': self.gpu, 'scientific': self.scientific,
                'phase_caps': {k: v.to_dict() for k, v in self.phase_caps.items()},
                'future': self.future.to_dict(), 'cleanup': self.cleanup.to_dict(),
                'deadline': self.deadline.isoformat()}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> StageLimits:
        if not isinstance(value, Mapping) or set(value) != {'gpu', 'scientific', 'phase_caps', 'future', 'cleanup', 'deadline'}:
            raise ValueError('invalid stage limits')
        return cls(value['gpu'], value['scientific'],
                   {PhaseName(k): Cost.from_dict(v) for k, v in value['phase_caps'].items()},
                   Cost.from_dict(value['future']), Cost.from_dict(value['cleanup']),
                   datetime.fromisoformat(value['deadline']))


@dataclass(frozen=True)
class StageRequest:
    """A fixed R38 operation; inputs[0] binds its operation-specific payload JSON."""
    name: str
    operation: Literal['prepare', 'preflight', 'measure', 'lock', 'fit', 'assess', 'verify']
    source: SourceIdentity
    inputs: tuple[ArtifactRef, ...]
    limits: StageLimits
    output: Path

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or re.fullmatch(r'[A-Za-z0-9_-]+', self.name) is None:
            raise ValueError('invalid stage name')
        if self.operation not in ('prepare', 'preflight', 'measure', 'lock', 'fit', 'assess', 'verify'):
            raise ValueError('unknown R38 operation')
        if not isinstance(self.source, SourceIdentity) or not isinstance(self.limits, StageLimits):
            raise ValueError('stage source and limits require domain values')
        if not self.inputs or any(not isinstance(ref, ArtifactRef) for ref in self.inputs):
            raise ValueError('stage requires bound operation payload and input references')
        object.__setattr__(self, 'inputs', tuple(self.inputs))
        object.__setattr__(self, 'output', Path(self.output))
        if not self.output.is_absolute():
            raise ValueError('stage output must be an absolute private path')
        if self.limits.scientific != (self.operation in ('fit', 'assess')):
            raise ValueError('scientific flag differs from fixed stage operation')

    def to_dict(self) -> dict[str, object]:
        return {'schema': 'r38-stage-request-v4', 'name': self.name, 'operation': self.operation,
                'implementation_source': self.source.to_dict(), 'inputs': [ref.to_dict() for ref in self.inputs],
                'limits': self.limits.to_dict(), 'output': str(self.output)}

    @classmethod
    def from_dict(cls, value: Mapping[str, object], *, repo: Path = REPO) -> StageRequest:
        if (not isinstance(value, Mapping)
                or set(value) != {'schema', 'name', 'operation', 'implementation_source', 'inputs', 'limits', 'output'}
                or value['schema'] != 'r38-stage-request-v4'):
            raise ValueError('invalid R38 stage request')
        return cls(value['name'], value['operation'], SourceIdentity.from_dict(value['implementation_source'], repo=repo),
                   tuple(ArtifactRef.from_dict(ref) for ref in value['inputs']),
                   StageLimits.from_dict(value['limits']), Path(value['output']))


def digest(path: str | Path) -> str:
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def save_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.pending')
    with temporary.open('w') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def week_order(visit: str) -> tuple[int, int]:
    match = re.fullmatch(r'week-(\d+)(?:-(\d+))?', visit)
    if match is None:
        raise ValueError(f'unrecognized encoded-week visit: {visit}')
    return int(match[1]), int(match[2] or 0)


def gap_weeks(previous: str, current: str) -> float:
    return float(week_order(current)[0] - week_order(previous)[0])


def source_hashes(repo: Path = REPO) -> dict[str, str]:
    paths = set(repo.glob('src/**/*.py')) | set(repo.glob('tests/test_joint*.py'))
    paths.add(repo / 'scripts/experiments/spatial/run_joint_dynamics.py')
    required = ('authority.py', 'baselines.py', 'assessment.py', 'protocol.py',
                'data.py', 'model.py', 'fitting.py', '__init__.py')
    for name in required:
        path = repo / 'src/harness/analysis/joint_dynamics' / name
        if not path.is_file():
            raise FileNotFoundError(f'incomplete implementation: {path}')
    return {str(path.relative_to(repo)): digest(path) for path in sorted(paths)}


def packet_identity(planning: Path = PLANNING) -> dict[str, str]:
    expected = {}
    for line in (planning / 'packet.sha256').read_text().splitlines():
        sha, name = line.split(maxsplit=1)
        name = name.strip()
        if Path(name).name != name or digest(planning / name) != sha:
            raise ValueError('planning packet bytes differ from packet.sha256')
        expected[name] = sha
    if not {'protocol-v3.md', 'numerical-interface-v3.md', 'input-bindings.json',
            'lane-board-v3.md'} <= expected.keys():
        raise ValueError('incomplete version-3 packet')
    expected['packet.sha256'] = digest(planning / 'packet.sha256')
    return expected


def implementation_identity(repo: Path = REPO) -> dict:
    return {'source_sha256': source_hashes(repo),
            'packet_sha256': packet_identity(repo / 'outputs/r38-joint-dynamics/planning'),
            'plan': json.loads(json.dumps(asdict(PLAN)))}


def verify_approval(approval: Mapping, identity: Mapping) -> None:
    if (approval.get('verdict') != 'APPROVED_FOR_R38_IMPLEMENTATION'
            or not approval.get('reviewer') or not approval.get('review_report_sha256')
            or approval.get('source_sha256') != identity['source_sha256']
            or approval.get('packet_sha256') != identity['packet_sha256']
            or approval.get('plan_digest') != canonical_hash(identity['plan'])):
        raise ValueError('independent stable implementation/protocol approval mismatch')


def file_manifest(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p.name != 'complete.json'}


def seal_stage(root: Path, kind: str, contract_digest: str, snapshot: dict) -> None:
    if (root / 'complete.json').exists():
        raise FileExistsError('completed artifact cannot be rewritten')
    save_json(root / 'complete.json', {'kind': kind, 'contract_digest': contract_digest,
              'files': file_manifest(root), 'authority': snapshot})


def verify_stage(root: Path, kind: str, contract_digest: str | None = None) -> dict:
    marker = read_json(root / 'complete.json')
    if (marker.get('kind') != kind or not marker.get('contract_digest')
            or (contract_digest is not None and marker['contract_digest'] != contract_digest)
            or marker.get('files') != file_manifest(root)):
        raise ValueError('incomplete, extra, changed or mismatched artifact files')
    return marker


def validate_completed_fit(root: Path, contract_digest: str) -> dict:
    marker = verify_stage(root, 'fit', contract_digest)
    fitted = read_json(root / 'fitted.json')
    expected = {f'{arm.name}_{seed}' for arm in ARMS for seed in PLAN.seeds}
    if (set(fitted.get('cells', {})) != expected or fitted.get('total_updates') != PLAN.total_updates
            or fitted.get('seeds') != list(PLAN.seeds)
            or fitted.get('contract_digest') != contract_digest):
        raise ValueError('incomplete or mismatched fixed matrix')
    for label, cell in fitted['cells'].items():
        if (cell.get('epoch') != 20 or cell.get('updates') != 5240
                or cell.get('contract_digest') != contract_digest):
            raise ValueError(f'incomplete cell: {label}')
        for key in ('weights', 'updates_file', 'calibration_file'):
            record = cell[key]
            if digest(root / record['path']) != record['sha256']:
                raise ValueError(f'changed fitted artifact: {label}/{key}')
    if fitted.get('prediction_sha256') != digest(root / 'private_predictions.npz'):
        raise ValueError('changed prediction bundle')
    return marker
