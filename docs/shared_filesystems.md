# Shared-filesystem artifact publication

Version 0.2.0rc13 supports managed record publication on filesystems that reject
hard links but provide the OS locks and atomic replacement required below.

Managed readers and writers coordinate through a per-artifact `.guard` file.
Exclusive publication first attempts a hard link. Only when the filesystem
returns EPERM, EOPNOTSUPP or ENOSYS does it fall back to checking that the target
is absent and renaming a complete regular staging file under that same exclusive
lock. Existing targets, including dangling symlinks, are refused. Other I/O
errors propagate. No empty target is reserved and no content is copied into a
visible target. A crash leaves the target absent or complete.

All writers of managed artifacts must obey this lock protocol. The fallback
does not promise OS-level no-replace behavior against external writers that
ignore the guard. A `.guard` file may persist after completion; its presence is
not evidence of a running job. The OS releases the lock on process exit.
Do not delete guards or edit managed records concurrently to bypass the protocol.

The installed rc13 candidate passed 556 full Linux regressions, 54 regressions
with test artifacts on an actual VirtualBox shared filesystem, and shared-folder
CLI initialization and numerical self-test. The Windows publisher-specific
suite passed 14 tests. Competing processes, readers, killed writers and
fail-closed errors are included. The first Linux run omitted an agent-card test
fixture; it was restored and the unchanged wheel passed the full rerun.

This qualifies the tested storage implementation, not every shared/NFS/SMB
filesystem, full Windows rc13 regression or a new scientific model. Qualify
locking, complete-file rename and collision/crash behavior before relying on
another filesystem. Numerical algorithms, knowledge, schemas and pinned vendors
are unchanged from rc12. Earlier scientific qualification keeps its exact
version and scope. This source update does not imply an external publication.

## rc16 guard initialization

The current implementation creates a missing `.guard` with exclusive creation
and opens an existing guard without creation flags. Concurrent creation through
ordinary append/open can produce different guest inode identities on an emulated
shared filesystem, defeating an otherwise valid per-inode lock. The exclusive
initializer prevents that creation race. The sidecar is a coordination marker;
the data target still appears only as a complete publication. No host-local lock
database, new hidden scientific state or change to dataset format is required.

Windows uses the native synchronous lock API: real contention waits and actual
permission errors propagate. It does not use the CRT's ten-retry cutoff or retry
every access denial. Existing sidecars remain usable and must not be deleted
while cooperating processes are active. Retain exact-version qualification
receipts; a finite successful test is not a guarantee for every filesystem or
for writers that ignore this protocol.
