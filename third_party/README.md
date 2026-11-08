# Document upstream model libraries, versions, licenses, and integration notes.

## Collection backend

`yam-abc-reproduce.lock.json` pins the Apache-2.0 collection fork and its I2RT
submodule. `python3 scripts/collect.py setup` installs the ignored checkout and
its separate environment. See [collection setup](../docs/collection.md).
