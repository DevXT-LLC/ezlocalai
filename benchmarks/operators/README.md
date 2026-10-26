# Environment recovery operator, 2026-10-26

`cache_env_recovery_20261026.py` preserves the reviewed V2 recovery helper for
an existing environment-update transaction. It is a host operator utility;
there is no installation, package integration or automatic execution. Its bytes
match the reviewed artifact:

```text
b5a56d8d924371d861a7205f04f103fac075b0565cbe6242eac22dbf94baec9f
```

Rollback is limited to the saved transaction and its recorded current container
identity. A later deployment with a different current container ID is unrelated
and causes refusal before cutover. This tool does not turn an old transaction
into a rollback plan for a newer deployment.

Before finalization, it restores the exact retained original container. After
finalization has removed that container, it recreates the original configuration
from the private inspection data, environment, dotenv backup and recorded
immutable image. The original `Config.Image` tag is explicitly normalized to
that immutable image ID, while all other Config, HostConfig, mounts, supported
network aliases and published ports are verified. Shared image aliases are not
retagged. Recreated containers have new IDs, which the result reports honestly.

The tuned container remains available until the recreated original is healthy
and fully verified. A failed recreation attempts to restore the tuned container
and pre-rollback dotenv. A verified original remains running across subsequent
cleanup errors, lost DELETE responses and state/proof write failures; retrying
uses its recorded ID.

## Recorded fleet transactions

The [RAM tuning report](../cache-depth-20261026.md#applied-ram-settings-and-rollback-materials)
records the applied `0.25` to `0.4` cache-fraction changes. These are the saved
transactions on the respective hosts:

| Worker | Host | Private run directory | Retained image tag |
| --- | --- | --- | --- |
| 4090 | `192.168.1.172` | `/tmp/ezlocalai-env-update-cache-env-20261026-devxt4090-c16abb26` | `ezlocalai:rollback-env-cache-env-20261026-devxt4090-c16abb26` |
| 3090V | `192.168.1.82` | `/tmp/ezlocalai-env-update-cache-env-20261026-devxt3090v-c16abb26` | `ezlocalai:rollback-env-cache-env-20261026-devxt3090v-c16abb26` |

Each directory contains `state.json`, `original-inspect.json` and
`dotenv.backup`; the backup is the original `/home/josh/ezlocalai/.env`. These
files contain private configuration and stay on their worker, with files at
mode `0600` inside a directory at mode `0700`. No private fleet state or
credentials are included in this repository.

## Explicit recovery workflow

The caller owns the router maintenance lease. Reserve the intended worker with
an owned idle drain, keep its lease renewed, and retain healthy alternate
workers. The helper itself does not acquire or release router leases.

Stage this helper on the selected host, for example as
`/tmp/cache_env_recovery_20261026.py`, then verify its SHA256 against the value
above. Run the matching command **on that worker** while its owned drain is
active. These commands are documentation, not evidence of an executed rollback.

For 4090:

```bash
python3 /tmp/cache_env_recovery_20261026.py rollback /tmp/ezlocalai-env-update-cache-env-20261026-devxt4090-c16abb26
```

For 3090V:

```bash
python3 /tmp/cache_env_recovery_20261026.py rollback /tmp/ezlocalai-env-update-cache-env-20261026-devxt3090v-c16abb26
```

Successful pre-finalization rollback preserves the original container ID.
Successful recreation reports `original_configuration_restored: true`,
`original_container_recreated: true`, `original_container_id_preserved: false`
and the actual `restored_container_id`. A nonzero exit requires inspecting the
private transaction/recovery result and live service before releasing the drain.

Verify container health, the effective engine arguments and dotenv, fresh router
metadata, and routed chat, streaming chat and vision pinned to the worker before
releasing the lease. Compare stable labels, models, capabilities, context and
GPU metadata; container and process IDs can change after recreation. Keep the
private backups and retained image tag.

Existing V1 transactions are supported without rewriting their private files
merely to stage this tool. They have no recorded original-inspect hash, so the
helper also binds their metadata to the original ID, immutable image, dotenv
hash, non-selected environment and current full configuration. Transactions
created through the retained V2 `apply` command additionally record an
original-inspect SHA256. The original `apply` and `finalize` commands remain in
this archived helper for compatibility; this recovery workflow uses `rollback`
on the existing saved transaction.

## Portable offline verification

From the checkout root:

```bash
python3 -m unittest discover -s benchmarks/operators -v
```

The test file can also run directly from any working directory. All 47 guards
use temporary files and synthetic Docker objects; real Docker requests are
blocked by the test harness. They cover existing V1 state, exact pre-finalization
IDs, finalized recreation, immutable image normalization with moved aliases,
configuration/network preservation, unrelated container refusal, bad backups
before stopping, failed-clone recovery, retained tuned containers, lost DELETE
responses, repeated rollback and post-verification state-write failures.

These tests and the independent review validate the operator logic. They do not
claim a live post-finalization fleet rollback, and adding this archive changes
no current worker setting.
