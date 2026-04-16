---
name: run-coral-ao-ut
description: Run torchAO unit tests on the Intel Coral (XPU) simulator environment. This skill covers environment setup, version validation of `torch` and `torchao`, optional package updates, and pytest execution.
---

# Skill: Run torchAO UT on Coral Simulator

## Purpose

Run torchAO unit tests on the Intel Coral (XPU) simulator environment. This skill covers environment setup, version validation of `torch` and `torchao`, optional package updates, and pytest execution.

---

## Key Paths

| Item | Path / URL |
|------|------------|
| Env script | `/home/sdp/xingyuan/env.sh` |
| torch-ao repo | Resolved at runtime via `pip show torchao` (see Step 5) |
| torch wheel index | `https://ubit-artifactory-ba.intel.com/artifactory/aipc_releases-ba-local/gpu-new/validation/JGS/` (requires `intel.com` in `no_proxy`) |

---

## Step 0 – Check for Running Python/Pytest Processes (MANDATORY)

> **NEVER operate on the Coral simulator itself.** Do NOT kill, restart, reconfigure, or otherwise touch any simulator/emulator processes (`coral`, `xesim`, `qemu`, or any process matching these names). The simulator is managed externally; killing it will make XPU unavailable (`sycl-ls` returns "No platforms found", `torch.xpu.is_available()` returns `False`) and **only an admin can restart it**. If the simulator appears broken, inform the user — do not attempt to fix it.

> **The Coral simulator only supports ONE Python process at a time.** Running a second Python/pytest process concurrently will cause hangs, crashes, or silent wrong results. **Always check before launching any test.**

```bash
ps aux | grep -E 'pytest|python.*test' | grep -v grep
```

- If any process running `pytest` or `python *test*` is listed, **wait for it to finish or kill it** before proceeding:
  ```bash
  pkill -u $USER -f "pytest|python.*test" 2>/dev/null
  ```
- **Do NOT blindly kill all `python` processes** — some belong to VS Code extensions (Pylance, language servers, etc.) and are unrelated to tests.
- Only proceed to the next step once no test-related Python processes are running.

---

## Step 1 – Source the Environment

```bash
source /home/sdp/xingyuan/env.sh
```

This sets up the Python environment and all required XPU/Coral/Triton/Inductor environment variables. Always run this first in any new terminal before executing tests or pip commands.

> **CRITICAL – `source env.sh` must be executed ALONE.**
> Never chain `source env.sh` with other commands using `&&` or `;` (e.g. `source env.sh && python foo.py`). The env script activates conda, sets `LD_LIBRARY_PATH`, and initializes the simulator connection — chaining it with subsequent commands can cause library resolution failures (`libsycl.so not found`) or environment variables not taking effect. Always run it as a standalone command first, then run subsequent commands separately in the same terminal.

> **CRITICAL – Always use the same terminal session:**
> All subsequent commands (version checks, pip installs, pytest runs) **must be executed in the same terminal shell** where `env.sh` was sourced. Never use a background terminal (`isBackground=true`) or spawn a new shell for test execution — new shells do not inherit the environment and `pytest` will not be found (system fallback only). Verify the environment is active before running tests:
> ```bash
> which pytest   # must return a path inside miniforge3/envs/...
> ```

---

## Step 2 – Check Package Versions

Run:

```bash
pip list 2>/dev/null | grep -E "^torch |^torchao "
```

This produces output like:

```
torch     2.12.0a0+git73a7172
torchao   0.17.0+git6506e361
```

---

## Step 3 – Validate torch Version (MANDATORY)

> **This step must never be skipped.** Always verify the torch version before running tests, even if torch was recently installed.

The latest torch wheel is published on the Intel Artifactory page:

```
https://ubit-artifactory-ba.intel.com/artifactory/aipc_releases-ba-local/gpu-new/validation/JGS/
```

> **Proxy note:** Add `intel.com` to `no_proxy` (or `NO_PROXY`) before accessing this URL.
> ```bash
> export no_proxy="${no_proxy},intel.com"
> ```

The directory is organised as `<year>/ww<number>/`, e.g. `2026/ww14/`. The wheel inside is named like:

```
torch-2.12.0a0+git73a7172-cp312-cp312-linux_x86_64.whl
```

### How to determine the latest version

1. List available `ww` subdirectories under the current year, then drill into the highest one:
```bash
export no_proxy="${no_proxy},intel.com"
YEAR=$(date +%Y)
curl -s "https://ubit-artifactory-ba.intel.com/artifactory/aipc_releases-ba-local/gpu-new/validation/JGS/${YEAR}/" \
    | grep -oP 'ww[0-9]+/' | sort -V | tail -1
# then list wheels in that directory, e.g.:
curl -s "https://ubit-artifactory-ba.intel.com/artifactory/aipc_releases-ba-local/gpu-new/validation/JGS/${YEAR}/ww14/" \
    | grep -oP 'torch-[^"]+\.whl'
```
2. Note the git hash embedded in the wheel filename (e.g. `git73a7172`).

### How to compare with the installed version

```bash
pip show torch 2>/dev/null | grep ^Version
# Example output: Version: 2.12.0a0+git73a7172
```

Compare the `+git<hash>` suffix with the hash in the latest wheel filename.  
If they differ, **torch is outdated** → proceed to Step 4a.  
If they match, torch is up to date → skip Step 4a.

---

## Step 4a – Update torch (if needed)

Download the latest wheel from the Artifactory page, then install it:

```bash
# Example – adjust URL and filename to the actual latest ww directory and hash
# Note: URL structure is <year>/<ww>/, e.g. 2026/ww14/
wget "https://ubit-artifactory-ba.intel.com/artifactory/aipc_releases-ba-local/gpu-new/validation/JGS/2026/ww14/torch-2.12.0a0+git73a7172-cp312-cp312-linux_x86_64.whl"
pip install --force-reinstall torch-2.12.0a0+git73a7172-cp312-cp312-linux_x86_64.whl
```

---

## Step 5 – Validate torchao Version

The installed `torchao` version embeds the short git commit hash (e.g. `0.17.0+git6506e361`). Compare it against the HEAD of the local repo:

```bash
AO_DIR=$(pip show torchao 2>/dev/null | grep ^Location | awk '{print $2}')
INST_AO_HASH=$(pip show torchao 2>/dev/null | grep ^Version | grep -oP '(?<=\+git)[0-9a-f]+')
HEAD_AO_HASH=$(git -C "$AO_DIR" log --format='%h' -1 HEAD)
echo "torch-ao repo  : $AO_DIR"
echo "Installed hash : $INST_AO_HASH"
echo "Repo HEAD hash : $HEAD_AO_HASH"
```

If `$INST_AO_HASH != $HEAD_AO_HASH`, **torchao is outdated** → proceed to Step 5a.  
If they match, torchao is up to date → skip Step 5a.

---

## Step 5a – Update torchao (if needed)

```bash
AO_DIR=$(pip show torchao 2>/dev/null | grep ^Location | awk '{print $2}')
cd "$AO_DIR"
PIP_NO_BUILD_ISOLATION=0 USE_CPP=0 pip install -e .
```

---

## Step 6 – Run UT

Before running tests, resolve the torch-ao repo root from the installed `torchao` package location (works regardless of which project directory is in use):

```bash
AO_REPO=$(pip show torchao 2>/dev/null | grep ^Location | awk '{print $2}')
echo "torch-ao repo: $AO_REPO"
cd "$AO_REPO"
```

The UT path is always relative to the repo root.

### Run a single test file

```bash
cd "$AO_REPO"
pytest -vs <test_file_path> --durations=0 -vv 2>&1 | tee /tmp/ut_run.log
```

**Examples:**

```bash
# Run all nvfp4 tensor tests
pytest -vs test/prototype/mx_formats/test_nvfp4_tensor.py --durations=0 -vv

# Run a specific test case
pytest -vs "test/prototype/mx_formats/test_nvfp4_tensor.py::test_nvfp4_reconstruction[xpu-block_scale_dtype0-dtype0-shape0-False]" --durations=0 -vv
```

### Run from a cases file (batch mode)

If a file listing test node-IDs exists (one per line, e.g. `ut_cases.txt`):

```bash
cd "$AO_REPO"
while IFS= read -r case; do
    pytest -vs "$case" --durations=0 -vv 2>&1 | tee -a /tmp/ut_run.log
done < /path/to/ut_cases.txt
```

---

## Step 7 – Batch UT with run_ut.sh (Recommended for Large Test Sets)

Due to Coral instability, individual tests frequently hang or trigger TBX errors. The helper script `/home/sdp/xingyuan/projects/20260302-nvfp4-enabling/run_ut.sh` handles per-case logging, TBX error detection, and automatic process killing.

### 7a – Collect test cases

> **IMPORTANT:** Always regenerate the cases list before each run. The UT file may have changed (new tests added, old tests removed, parametrize values updated) since the list was last generated. Never reuse a stale `ut_cases.txt`.

Run `pytest --collect-only` to generate the cases list, then trim the trailing summary lines with `head -n -2`:

```bash
cd "$AO_REPO"
pytest --collect-only -q <test_target> | head -n -2 > ut_cases.txt
```

`<test_target>` can be a test file, directory, or `-k` expression. Example:

```bash
pytest --collect-only -q test/prototype/mx_formats/test_nvfp4_tensor.py -k "xpu" | head -n -2 > ut_cases.txt
```

#### Known slow tests to skip on Coral

`test_nvfp4_slicing` runs **extremely slowly** on the Coral simulator and should be excluded.

`test_nvfp4_matmul_with_amax` cases with `use_triton_kernel=False` + `quant_type=dynamic` always **timeout (3600s)** on the Coral simulator. The non-Triton kernel path for dynamic quantization is not functional on XPU. Exclude them by deselecting `dynamic` (which removes all `-dynamic-` parametrized IDs when `use_triton_kernel=False`, i.e. the slow ones) — or more precisely, deselect the combination with `-k "not (dynamic and not use_triton)"`. The safest practical filter is to **skip all `dynamic` cases for `test_nvfp4_matmul_with_amax`** when running locally:

```bash
pytest --collect-only -q test/prototype/mx_formats/test_nvfp4_tensor.py \
    -k "xpu and not test_nvfp4_slicing and not (test_nvfp4_matmul_with_amax and dynamic)" \
    | head -n -2 > ut_cases.txt
```

> **Why skip all `dynamic` in `test_nvfp4_matmul_with_amax`?** Analysis of log `nvfp4p-ut-log/20260409-1352/nohup.log` shows every `use_triton_kernel=False + dynamic` combination hits 3600s timeout. The `use_triton_kernel=True + dynamic` cases complete in ~8s, but they are SKIPped (not actually running on Coral anyway). Filtering out the whole `dynamic` variants avoids the timeouts with no loss of coverage.

### 7b – Run with run_ut.sh

Name the log directory with minute-level precision so multiple runs on the same day are easy to distinguish:

```bash
LOG_DIR=/home/sdp/xingyuan/projects/20260302-nvfp4-enabling/nvfp4p-ut-log/$(date +%Y%m%d-%H%M)
mkdir -p "$LOG_DIR"
nohup bash /home/sdp/xingyuan/projects/20260302-nvfp4-enabling/run_ut.sh \
    -f /home/sdp/xingyuan/projects/20260302-nvfp4-enabling/ut_cases.txt \
    -l "$LOG_DIR" \
    -w "$AO_REPO" \
    > "$LOG_DIR/nohup.log" 2>&1 &
UT_PID=$!
echo "Started PID=$UT_PID — logs: $LOG_DIR"
```

Monitor progress:
```bash
tail -f "$LOG_DIR/nohup.log"
```

- `-f` — path to the cases file produced in step 7a  
- `-l` — output directory (`$(date +%Y%m%d-%H%M)` gives e.g. `20260409-1111`); a master log `<log_dir>/<log_dir_basename>.log` and per-case logs are written here  
- `-w` — working directory (torch-ao repo root)

---

## Coral Operational Notes

### Ctrl+C does not stop Python processes on Coral

Python programs running on the Coral simulator typically **do not respond to Ctrl+C**. To stop the run, use `$UT_PID` saved at launch (Step 7b). Send `SIGTERM` first and only escalate to `SIGKILL` if needed:

```bash
# Try graceful termination first
kill $UT_PID
sleep 5

# If still alive, kill the full process tree
pstree -pT $UT_PID | grep -oP '\(\K[0-9]+(?=\))' | sort -rV | xargs kill -9
```

If `$UT_PID` is no longer in scope (e.g. in a new shell), fall back to pgrep:
```bash
pgrep -u "$USER" -a | grep run_ut
# then use the PID found above in place of $UT_PID
```

> Use `-9` only as a last resort. Prefer `SIGTERM` first to allow cleanup.

### `kill $UT_PID` alone is not enough

`run_ut.sh` spawns a **process tree**: the shell → `timeout` → `pytest` → `python3.12`. When the parent shell exits, child processes (`timeout`, `pytest`, `python3.12`) **continue running** as orphans. Always kill the entire tree after killing the parent:

```bash
# Step 1: kill parent
kill $UT_PID
sleep 3

# Step 2: kill all orphaned children by iterating over every known PID
for pid in $(pgrep -u "$USER" -a | grep -E "pytest|run_ut" | awk '{print $1}'); do
    pstree -pT $pid 2>/dev/null | grep -oP '\(\K[0-9]+(?=\))' | sort -rn | xargs -r kill -9 2>/dev/null
    kill -9 $pid 2>/dev/null
done

# Step 3: verify clean
pgrep -u "$USER" -a | grep -E "pytest|run_ut" | grep -v grep
```

If `pstree` is not available or returns nothing (process already exited), the direct `kill -9 $pid` in the loop still handles that PID.

### TBX error = Coral disconnected

When the output contains `TbxSocketsImp Error`, the test process has lost its connection to the Coral simulator and will never complete. Kill the process immediately to save time. Use `$UT_PID` and the same two-step approach:

```bash
kill $UT_PID
sleep 5
pstree -pT $UT_PID | grep -oP '\(\K[0-9]+(?=\))' | sort -rV | xargs kill -9
```

> `run_ut.sh` monitors for this error automatically and kills the process when detected.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'torchao'` | Reinstall torchao (Step 5a) |
| `SYCL` / driver errors on XPU ops | Confirm `source /home/sdp/xingyuan/env.sh` was executed in the current shell |
| `FsScheduler coral error` | Ensure `DeferStateInitSubmissionToFirstRegularUsage=1` is exported (set by env.sh) |
| Triton compile failures | Verify `TRITON_INTEL_PISA_COMPILER=triton`, `TRITON_XPU_GEN_NATIVE_CODE=0`, `TRITON_INTEL_FORCE_DISABLE_WRAPPERS=1` are all set (set by env.sh) |
| Test hangs / multiprocess deadlock | Verify `TORCHINDUCTOR_WORKER_START=fork` and `TORCHINDUCTOR_COMPILE_THREADS=1` are set |
| Test hangs and Ctrl+C not working | Kill the process tree manually (see Coral Operational Notes above) |
| Output contains `TbxSocketsImp Error` | Coral disconnected — kill the process immediately to save time |
