import re

REPO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def validate_repo_id(repo_id: str) -> str:
    """Ensure repo_id is safe to use as a filesystem path component.

    repo_id is interpolated directly into paths like `data/indexes/{repo_id}.index`.
    Without this check, a value like '../../etc/passwd' could escape the data
    directory (path traversal). Only letters, digits, '_' and '-' are allowed.
    """
    if not REPO_ID_PATTERN.match(repo_id):
        raise ValueError(
            f"Invalid repo_id '{repo_id}': must be 1-128 characters of "
            f"letters, digits, '_' or '-'."
        )
    return repo_id
