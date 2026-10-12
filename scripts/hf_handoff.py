"""Package, publish, download and checksum the public training handoff.

Never packages .env, Git metadata, runs, crawler trash or credentials.
The existing training/pipeline files are deliberately left untouched.
"""
import argparse
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import zipfile

# The Windows Xet backend can fail with os error 183 and retry indefinitely.
# Set before importing huggingface_hub; an explicit caller setting wins.
if os.name == 'nt':
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

ROOT = Path(__file__).resolve().parents[1]
READY = 'DataProcessed/pretrain_ready/'
TRAIN_PATTERNS = [READY + name + '/*' for name in
                  ('binary', 'tokenizer', 'clean', 'checkpoints', 'reports')]
SECRET = re.compile(rb'hf_[A-Za-z0-9]{20,}')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def load_token(root=ROOT):
    """Read only supported token names; never echo the value or mutate the environment."""
    if os.environ.get('HF_TOKEN'):
        return os.environ['HF_TOKEN']
    env = Path(root) / '.env'
    if env.exists():
        values = {}
        for line in env.read_text(encoding='utf-8-sig').splitlines():
            if '=' not in line or line.lstrip().startswith('#'):
                continue
            key, value = line.split('=', 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
        return values.get('HF_TOKEN') or values.get('hugging_key')
    return None  # The Hub may use the receiver's saved login; public download needs none.


def safe_path(name):
    parts = PurePosixPath(name)
    if (parts.is_absolute() or not parts.parts or '..' in parts.parts or
            '\\' in name or ':' in name):
        raise ValueError(f'Unsafe manifest path: {name}')
    return parts


def excluded(path):
    return (any(part in ('.git', '.cache', '__pycache__', '.venv', 'venv') for part in path.parts)
            or path.name.startswith('.env') or path.suffix in ('.pyc', '.pyo', '.tmp', '.lock')
            or path.name.endswith(('-wal', '-shm', '.journal')))


def source_files(root):
    files = []
    for name in ('scripts', 'configs'):
        files.extend(p for p in (root / name).rglob('*') if p.is_file() and not excluded(p))
    for name in ('README.md', '.gitignore', '.gitattributes', 'pipeline_pretrain_llm_tieng_viet.txt'):
        path = root / name
        if path.is_file():
            files.append(path)
    return sorted(files)


def selected(name, profile):
    return profile == 'all' or any(fnmatch.fnmatchcase(name, pattern) for pattern in TRAIN_PATTERNS)


def package(root, stage):
    root, stage = Path(root).resolve(), Path(stage).resolve()
    if stage == root or root not in stage.parents or not stage.is_relative_to(root / 'runs'):
        raise ValueError('Staging must be inside the project runs directory')
    manifest = read_json(root / READY / 'binary/manifest.json')
    if manifest.get('complete') is not True:
        raise RuntimeError('Refusing to publish incomplete training data')
    if read_json(root / READY / 'reports/verification.json').get('passed') is not True:
        raise RuntimeError('Pipeline verification must pass before packaging')
    stage.mkdir(parents=True, exist_ok=True)
    sources = source_files(root)
    data = sorted(p for folder in ('DataProcessed', 'DataRaw')
                  for p in (root / folder).rglob('*') if p.is_file() and not excluded(p))
    if any(p.is_symlink() for p in sources + data):
        raise RuntimeError('Symlinks are not allowed in a handoff')
    for path in sources + data:
        if path.stat().st_size <= 2 * 1024 * 1024 and SECRET.search(path.read_bytes()):
            raise RuntimeError(f'Possible Hugging Face credential in {path.relative_to(root)}; not published')
    with zipfile.ZipFile(stage / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path, path.relative_to(root).as_posix())
    # Hardlinks avoid another 28 GiB local copy. Do not modify the source data during upload.
    expected = {'source.zip', 'README.md', 'handoff-manifest.json'}
    entries = []
    for index, path in enumerate(data, 1):
        name = path.relative_to(root).as_posix()
        expected.add(name)
        target = stage / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        try:
            os.link(path, target)
        except OSError:
            shutil.copy2(path, target)
        entries.append({'path': name, 'bytes': path.stat().st_size, 'sha256': sha256(target)})
        if index % 500 == 0:
            print(f'Packaged and hashed {index}/{len(data)} data files', flush=True)
    # Remove stale staged files, but preserve resumable Hub upload metadata.
    for path in stage.rglob('*'):
        if path.is_file() and '.cache' not in path.relative_to(stage).parts:
            if path.relative_to(stage).as_posix() not in expected:
                path.unlink()
    source = stage / 'source.zip'
    entries.append({'path': 'source.zip', 'bytes': source.stat().st_size, 'sha256': sha256(source)})
    config = read_json(root / 'configs/handoff.json')
    write_json(stage / 'handoff-manifest.json', {
        'format_version': 1, 'repo_id': config['repo_id'], 'revision_tag': config['revision'],
        'created_at': datetime.now(timezone.utc).isoformat(),
        'training_data_complete': True, 'raw_corpus_complete': False,
        'files': entries, 'total_bytes': sum(entry['bytes'] for entry in entries),
        'source_files': [{'path': p.relative_to(root).as_posix(), 'sha256': sha256(p)} for p in sources],
        'profiles': {'train': TRAIN_PATTERNS, 'all': 'All files listed in this manifest'},
    })
    card = ('---\nlanguage:\n- vi\npretty_name: AgentContentVietnamese training handoff\n'
            'size_categories:\n- 1M<n<10M\ntask_categories:\n- text-generation\n'
            'tags:\n- pretraining\n- vietnamese\n---\n\n')
    (stage / 'README.md').write_text(card + (root / 'README.md').read_text(encoding='utf-8'), encoding='utf-8')
    print(f'Package ready: {len(entries)} files, {sum(e["bytes"] for e in entries) / 1024**3:.3f} GiB', flush=True)


def verify_files(directory, manifest, profile='all'):
    checked, total = 0, 0
    for entry in manifest['files']:
        name = entry['path']
        safe_path(name)
        if not selected(name, profile):
            continue
        path = Path(directory) / name
        if not path.is_file() or path.stat().st_size != entry['bytes']:
            raise RuntimeError(f'Missing file or size mismatch: {name}')
        if sha256(path) != entry['sha256']:
            raise RuntimeError(f'SHA-256 mismatch: {name}')
        checked += 1
        total += entry['bytes']
    if not checked:
        raise RuntimeError('No files selected for verification')
    result = {'passed': True, 'profile': profile, 'checked_files': checked, 'checked_bytes': total}
    print(json.dumps(result), flush=True)
    return result


def verify_remote(api, repo, revision, manifest):
    """Compare every remote file size and every available LFS SHA-256 (no weights loaded)."""
    remote = {item.path: item for item in api.list_repo_tree(repo, repo_type='dataset',
              revision=revision, recursive=True) if hasattr(item, 'size')}
    lfs_checks = 0
    for entry in manifest['files']:
        item = remote.get(entry['path'])
        if item is None or item.size != entry['bytes']:
            raise RuntimeError(f'Remote file missing or size differs: {entry["path"]}')
        if item.lfs:
            if item.lfs.sha256 != entry['sha256']:
                raise RuntimeError(f'Remote LFS checksum differs: {entry["path"]}')
            lfs_checks += 1
    return {'remote_files_checked': len(manifest['files']), 'remote_lfs_sha256_checked': lfs_checks}


def upload(args, config):
    from huggingface_hub import HfApi
    token = load_token()
    if not token:
        raise RuntimeError('Set HF_TOKEN or hugging_key in the local .env before publishing')
    api = HfApi(token=token)
    print(f'Authenticated Hugging Face account: {api.whoami()["name"]}', flush=True)
    # Never turn an existing private repository public without explicit approval.
    api.create_repo(config['repo_id'], repo_type='dataset', private=False, exist_ok=True)
    if api.repo_info(config['repo_id'], repo_type='dataset').private:
        raise RuntimeError('Existing repository is private; visibility must be reviewed explicitly')
    api.upload_large_folder(config['repo_id'], folder_path=args.stage, repo_type='dataset',
                            num_workers=args.workers, ignore_patterns=['.cache/**'], print_report_every=60)
    commit = api.repo_info(config['repo_id'], repo_type='dataset').sha
    checks = verify_remote(api, config['repo_id'], commit, read_json(args.stage / 'handoff-manifest.json'))
    # A fixed tag pins the entire source+data snapshot. Never move an existing release tag.
    refs = api.list_repo_refs(config['repo_id'], repo_type='dataset')
    existing = next((tag for tag in refs.tags if tag.name == config['revision']), None)
    if existing and existing.target_commit != commit:
        raise RuntimeError('Release tag already points elsewhere; choose a new revision in handoff.json')
    if not existing:
        api.create_tag(config['repo_id'], tag=config['revision'], revision=commit, repo_type='dataset')
    receipt = dict(repo_id=config['repo_id'], revision=config['revision'], commit=commit,
                   public=True, completed_at=datetime.now(timezone.utc).isoformat(), **checks)
    write_json(ROOT / 'runs/hf_handoff/upload_receipt.json', receipt)
    print(json.dumps(receipt, indent=2), flush=True)


def download(args, config):
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    repo, revision = args.repo_id or config['repo_id'], args.revision or config['revision']
    token = load_token()
    # Resolve once so a concurrent update cannot mix different revisions.
    commit = HfApi(token=token).repo_info(repo, repo_type='dataset', revision=revision).sha
    args.destination.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(repo, 'handoff-manifest.json', repo_type='dataset', revision=commit,
                           local_dir=args.destination, token=token)
    manifest = read_json(path)
    for entry in manifest['files']:
        safe_path(entry['path'])
    patterns = None if args.profile == 'all' else TRAIN_PATTERNS + ['handoff-manifest.json', 'README.md']
    snapshot_download(repo, repo_type='dataset', revision=commit, local_dir=args.destination,
                      token=token, allow_patterns=patterns, max_workers=args.workers)
    result = verify_files(args.destination, manifest, args.profile)
    write_json(args.destination / '.cache/handoff-download-receipt.json',
               dict(repo_id=repo, requested_revision=revision, commit=commit, **result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    default_stage = ROOT / 'runs/hf_handoff/staging'
    pack = sub.add_parser('package')
    pack.add_argument('--stage', type=Path, default=default_stage)
    publish = sub.add_parser('upload')
    publish.add_argument('--stage', type=Path, default=default_stage)
    publish.add_argument('--workers', type=int, default=4)
    receive = sub.add_parser('download')
    receive.add_argument('--destination', type=Path, default=ROOT)
    receive.add_argument('--profile', choices=('train', 'all'), default='all')
    receive.add_argument('--repo-id')
    receive.add_argument('--revision')
    receive.add_argument('--workers', type=int, default=4)
    check = sub.add_parser('verify')
    check.add_argument('--destination', type=Path, default=ROOT)
    check.add_argument('--profile', choices=('train', 'all'), default='all')
    args = parser.parse_args()
    if getattr(args, 'workers', 1) < 1:
        parser.error('--workers must be positive')
    config = read_json(ROOT / 'configs/handoff.json')
    if args.command == 'package':
        package(ROOT, args.stage)
    elif args.command == 'upload':
        upload(args, config)
    elif args.command == 'download':
        download(args, config)
    else:
        verify_files(args.destination, read_json(args.destination / 'handoff-manifest.json'), args.profile)


if __name__ == '__main__':
    main()