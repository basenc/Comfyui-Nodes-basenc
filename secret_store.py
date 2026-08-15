import json
import os
import tempfile
from pathlib import Path
from urllib.parse import quote, urlparse
from uuid import uuid4

import folder_paths
from cryptography.fernet import Fernet

SECRET_SCHEME = "basenc-secret"


def _store_directory() -> Path:
    directory = Path(folder_paths.get_system_user_directory("basenc"))
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory.chmod(0o700)
    return directory


def _write_private(path: Path, content: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
        file.write(content)
        file.flush()
        os.fsync(file.fileno())
    Path(file.name).chmod(0o600)
    os.replace(file.name, path)


def _fernet() -> Fernet:
    path = _store_directory() / "api_keys.key"
    if not path.exists():
        _write_private(path, Fernet.generate_key())
    return Fernet(path.read_bytes())


def _read_secrets() -> dict[str, str]:
    path = _store_directory() / "api_keys.enc"
    if not path.exists():
        return {}
    return json.loads(_fernet().decrypt(path.read_bytes()))


def _masked(secret: str) -> str:
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:3]}{'*' * 8}{secret[-4:]}"


def store_secret(secret: str) -> str:
    if not secret:
        raise ValueError("API key cannot be empty.")
    identifier = uuid4().hex
    secrets = _read_secrets()
    secrets[identifier] = secret
    _write_private(
        _store_directory() / "api_keys.enc",
        _fernet().encrypt(json.dumps(secrets).encode()),
    )
    return f"{SECRET_SCHEME}://{identifier}#{quote(_masked(secret), safe='*')}"


def resolve_secret(value: str) -> str:
    reference = urlparse(value)
    if reference.scheme != SECRET_SCHEME:
        return value
    secret = _read_secrets().get(reference.netloc)
    if secret is None:
        raise ValueError("The stored API key is unavailable on this ComfyUI instance.")
    return secret
