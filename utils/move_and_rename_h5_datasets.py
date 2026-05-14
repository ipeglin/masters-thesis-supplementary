import os
from dataclasses import dataclass

import h5py


# Configure your migration here.
# Supported action values: "move", "delete", "copy_attributes"
# - move: moves either a dataset or a group from source to destination
# - delete: deletes the target if it exists
# - copy_attributes: copies attributes from source to destination
@dataclass
class MigrationOperation:
    action: str
    source: str | None = None
    destination: str | None = None
    target: str | None = None
    create_parents: bool = True
    overwrite_destination: bool = False
    preserve_attributes: bool = True
    recursive: bool = True
    overwrite_attributes: bool = True


MIGRATION_OPERATIONS = [
    # MigrationOperation(
    #     action="move",
    #     source="04mvmd/",
    #     destination="04mvmd_bak/",
    # ),
    # Example attribute-only copy:
    # MigrationOperation(
    #     action="copy_attributes",
    #     source="mvmd_rest/",
    #     destination="mvmd/",
    #     recursive=True,
    #     overwrite_attributes=True,
    # ),
    # Example cleanup:
    MigrationOperation(action="delete", target="/01fmri_parcellation/full_run_raw"),
    # MigrationOperation(action="delete", target="/04mvmd/full_run_std"),
]


def _normalize_h5_path(path: str) -> str:
    return path.strip("/")


def _ensure_parent_group(h5_file: h5py.File, path: str) -> None:
    normalized_path = _normalize_h5_path(path)
    if not normalized_path or "/" not in normalized_path:
        return

    parent_path = normalized_path.rsplit("/", 1)[0]
    h5_file.require_group(parent_path)


def _clear_attributes_recursive(h5_obj) -> None:
    for key in list(h5_obj.attrs.keys()):
        del h5_obj.attrs[key]

    if isinstance(h5_obj, h5py.Group):
        for child_name in h5_obj.keys():
            _clear_attributes_recursive(h5_obj[child_name])


def _collect_attributes_recursive(h5_obj, base_path: str = "") -> dict[str, dict[str, object]]:
    collected: dict[str, dict[str, object]] = {
        base_path: {key: h5_obj.attrs[key] for key in h5_obj.attrs.keys()}
    }

    if isinstance(h5_obj, h5py.Group):
        for child_name in h5_obj.keys():
            child_path = f"{base_path}/{child_name}" if base_path else child_name
            collected.update(_collect_attributes_recursive(h5_obj[child_name], child_path))

    return collected


def _apply_attributes_snapshot(
    destination_obj,
    attributes_snapshot: dict[str, dict[str, object]],
    overwrite_attributes: bool,
) -> None:
    for relative_path, attrs in attributes_snapshot.items():
        target_obj = destination_obj if not relative_path else destination_obj[relative_path]

        for key, value in attrs.items():
            if key in target_obj.attrs and not overwrite_attributes:
                continue
            target_obj.attrs[key] = value


def _copy_attributes(
    source_obj,
    destination_obj,
    recursive: bool,
    overwrite_attributes: bool,
) -> None:
    if recursive:
        attrs_snapshot = _collect_attributes_recursive(source_obj)
        _apply_attributes_snapshot(destination_obj, attrs_snapshot, overwrite_attributes)
        return

    for key in source_obj.attrs.keys():
        if key in destination_obj.attrs and not overwrite_attributes:
            continue
        destination_obj.attrs[key] = source_obj.attrs[key]


def _apply_move(h5_file: h5py.File, operation: MigrationOperation) -> str:
    if not operation.source or not operation.destination:
        return "Skipped move: source/destination not provided"

    source = _normalize_h5_path(operation.source)
    destination = _normalize_h5_path(operation.destination)

    if source not in h5_file:
        return f"Skipped move: source not found ({source})"

    if operation.create_parents:
        _ensure_parent_group(h5_file, destination)

    if destination in h5_file:
        if not operation.overwrite_destination:
            return f"Skipped move: destination exists ({destination})"
        del h5_file[destination]

    attrs_snapshot = None
    if operation.preserve_attributes:
        attrs_snapshot = _collect_attributes_recursive(h5_file[source])

    h5_file.move(source, destination)

    if operation.preserve_attributes and attrs_snapshot is not None:
        _apply_attributes_snapshot(
            h5_file[destination],
            attrs_snapshot,
            overwrite_attributes=True,
        )
    else:
        _clear_attributes_recursive(h5_file[destination])

    return f"Moved {source} -> {destination}"


def _apply_delete(h5_file: h5py.File, operation: MigrationOperation) -> str:
    if not operation.target:
        return "Skipped delete: target not provided"

    target = _normalize_h5_path(operation.target)

    if target not in h5_file:
        return f"Skipped delete: target not found ({target})"

    del h5_file[target]
    return f"Deleted {target}"


def _apply_copy_attributes(h5_file: h5py.File, operation: MigrationOperation) -> str:
    if not operation.source or not operation.destination:
        return "Skipped copy_attributes: source/destination not provided"

    source = _normalize_h5_path(operation.source)
    destination = _normalize_h5_path(operation.destination)

    if source not in h5_file:
        return f"Skipped copy_attributes: source not found ({source})"

    if destination not in h5_file:
        return f"Skipped copy_attributes: destination not found ({destination})"

    _copy_attributes(
        h5_file[source],
        h5_file[destination],
        recursive=operation.recursive,
        overwrite_attributes=operation.overwrite_attributes,
    )
    recursive_note = "recursive" if operation.recursive else "non-recursive"
    return f"Copied attributes ({recursive_note}) {source} -> {destination}"


def apply_operations_to_file(filepath: str, operations: list[MigrationOperation]) -> list[str]:
    messages: list[str] = []

    with h5py.File(filepath, "a") as h5_file:
        for operation in operations:
            action = operation.action.lower().strip()

            if action == "move":
                messages.append(_apply_move(h5_file, operation))
            elif action == "delete":
                messages.append(_apply_delete(h5_file, operation))
            elif action == "copy_attributes":
                messages.append(_apply_copy_attributes(h5_file, operation))
            else:
                messages.append(f"Skipped: unsupported action ({operation.action})")

    return messages


def migrate_bids_data(root_dir: str, operations: list[MigrationOperation]) -> None:
    all_h5_files = []
    for root, _dirs, files in os.walk(root_dir):
        for f in files:
            if f.startswith("sub-") and f.endswith(".h5"):
                all_h5_files.append((root, f))
    
    total_files = len(all_h5_files)
    print(f"Found {total_files} matching HDF5 files to process.")

    for i, (root, filename) in enumerate(all_h5_files, 1):
        filepath = os.path.join(root, filename)
        relative_dir = os.path.relpath(root, root_dir)
        print(f"[{i}/{total_files}] Processing {relative_dir}/{filename} ...")

        try:
            file_messages = apply_operations_to_file(filepath, operations)
            for message in file_messages:
                print(f"  -> {message}")
        except Exception as error:
            print(f"  -> Error: {error}")


if __name__ == "__main__":
    ROOT_DIR = "/Users/ipeglin/Documents/masters_thesis/bids_processed_consolidated_data"
    # ROOT_DIR = "/Volumes/work/bids_processed_consolidated_data" # IDUN Network mount
    migrate_bids_data(ROOT_DIR, MIGRATION_OPERATIONS)
