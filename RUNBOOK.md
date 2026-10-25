# Runbook: final numbers on the fixed machine

The machine: a laptop with a Ryzen 7, an RTX 3050 Laptop GPU and 16 GB RAM, running Windows with Ubuntu 24.04 in WSL2. Every final number in the report comes from this one machine (DESIGN §10, BEN-4).

Run the steps in order. Each ends with what to check before moving on. I don't know how long the runs take on this machine yet; plan for a whole day.

---

## 0. Windows, once

1. Update the NVIDIA driver from nvidia.com. The Windows driver is all WSL2 needs: never install an NVIDIA driver inside Ubuntu.
2. In PowerShell as administrator:
   ```powershell
   wsl --install -d Ubuntu-24.04
   ```
   Reboot, open "Ubuntu 24.04" from the Start menu, and create the Linux user.
3. Power settings (Settings → System → Power & battery): plugged in, power mode **Best performance**, screen and sleep **Never**, both plugged in and on battery (the battery-saver run happens unplugged).

## 1. Ubuntu, once

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip git stress-ng tmux zip
```
```bash
nvidia-smi
```
**Check:** `nvidia-smi` shows the RTX 3050 and its memory (4 GB or 6 GB). Write the memory down; step 9 depends on it.

## 2. The code

Clone into the Linux home folder, not `/mnt/c`: files on the Windows drive are slow from WSL.
```bash
cd ~ && git clone https://github.com/VARITHSA/autohetero.git && cd autohetero
```
```bash
git log -1 --oneline
```
```bash
python3 -m venv .venv && source .venv/bin/activate
```
```bash
pip install torch torchvision
```
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
```
```bash
pip install -e ".[dev]"
```
**Check:** the torch line prints `True` and the RTX 3050. If it prints `False`, install the CUDA build with the command from pytorch.org. Write down the commit `git log` printed: every number comes from that commit.

## 3. Tests

```bash
pytest
```
```bash
pytest -m gpu 2>&1 | tee gpu-tests.txt
```
**Check:** everything passes, including the 44 `gpu` tests (the weeks 1–2 parity check and the week-6 switch check). If anything fails, stop here: no measurement is worth taking on code that fails its tests. Out-of-memory failures matter too, since the card is small.

## 4. Record the machine

```bash
mkdir -p eval/out && nvidia-smi > eval/out/machine-gpu.txt && lscpu > eval/out/machine-cpu.txt && free -h > eval/out/machine-ram.txt
```

## 5. Calibrate once

```bash
python -c "import autohetero; autohetero.enable()"
```
```bash
ls ~/.cache/autohetero/
```
**Check:** calibration prints its progress and finishes in about 30 s. `ls` shows one folder (the machine key); its `timings.csv` is used in step 7.

## 6. Work inside tmux

A long run must survive a closed terminal.
```bash
tmux new -s ah
```
Inside it, `cd ~/autohetero && source .venv/bin/activate`. To come back later: `tmux attach -t ah`. From here on, close every Windows app and don't use the laptop while it measures.

## 7. Dataset, crossover, RQ1

```bash
python bench/collect.py
```
```bash
python eval/crossover.py
```
```bash
python eval/rq1.py bench/data/<dataset>.csv --calibration ~/.cache/autohetero/<key>/timings.csv
```
**Check:** `bench/data/` has one `.csv` and one `.json`. `eval/out/` has `crossover.png`, `crossover.csv`, `rq1.csv`, `rq1_by_op.csv`. This is the weeks 3–4 exit check, plus RQ1.

## 8. Checkpoint: tune, or don't

The executor constants (`SLOWDOWN`, `FIRST_CHUNK`, `MIN_CHUNK_SECONDS`, `HYBRID_ROUND`, `HYBRID_MARGIN`) are first guesses, and finding 3 (built-in ops ignore load) is undecided (REQUIREMENTS, "Findings").
- **Tune:** copy `bench/data/` back to the Mac, tune and commit there, push, then `git pull` here and repeat step 3 before going on.
- **Don't tune:** go on, and say in the report that the constants are untuned first guesses.

Either way, anything changed after step 9 means running step 9 again.

## 9. Experiments

**GPU memory.** On a 4 GB card, the largest RQ2 image workloads (64 × 3 × 1080 × 1920) don't fit. RQ2 checks every workload against free GPU memory under each condition and skips the ones that don't fit, printing `skipped: too big for free GPU memory`. A workload can fit at idle but not under GPU stress, since the stress job holds memory too. Write down every skipped line: the report names them as a limit of the card.

RQ2: idle, GPU stress and CPU stress plugged in, battery saver last. When it asks, unplug the charger, turn on battery saver (Settings → System → Power & battery), and press Enter.
```bash
python bench/experiments.py rq2 --conditions idle,gpu,cpu,battery
```
Then plug back in, turn battery saver off, set Best performance again, and run Hybrid gain:
```bash
python bench/experiments.py hybrid-gain
```
**Check:** `bench/results/` has two `.csv` files, each with a `.json`. The CPU-stress line printed `cpu stress: stress-ng`.

## 10. Check fallbacks

The memory check estimates; it can't see every intermediate a workload allocates. `fallback` is the only true/false column in the results, so this counts runs where the GPU failed anyway and the CPU stood in. After one failure, every later forced CPU / GPU run of that workload size in the same experiment run goes straight to the CPU, and each of those counts too:
```bash
grep -c ",True," bench/results/*.csv
```
**Check:** 0 for both files. Anything else: those workloads' GPU-only and AutoHetero numbers aren't GPU numbers. Leave them out of the report, or name them as a limit of the card.

## 11. The rest of the evaluation

Give each script its own results file (by default they take the newest one):
```bash
python eval/overhead.py
```
```bash
python eval/rq2.py bench/results/<rq2 file>.csv
```
```bash
python eval/hybrid_gain.py bench/results/<hybrid-gain file>.csv
```
```bash
python eval/rq3.py
```
**Check:** `eval/out/` now has 12 result files: `crossover.png/.csv`, `rq1.csv`, `rq1_by_op.csv`, `rq2.csv`, `rq2_summary.csv`, `rq2.png`, `hybrid_gain.csv`, `hybrid_gain_summary.csv`, `hybrid_gain.png`, `overhead.csv`, `rq3.csv`. Read every "skipped" line the scripts printed. If a result looks noisy (high spread across runs), rerun that experiment with `--reps 10`.

## 12. Bring the results back

```bash
cd ~ && zip -r autohetero-results.zip autohetero/bench/data autohetero/bench/results autohetero/eval/out autohetero/gpu-tests.txt .cache/autohetero
```
In Windows Explorer, open `\\wsl$\Ubuntu-24.04\home\<user>\` and copy `autohetero-results.zip` to the Mac.

Then, on the Mac: fill the pending cells in CARRYOVER.md, the status in REQUIREMENTS.md, and the report and deck, all from `eval/out/` only (EVL-7). Tag `autohetero-v1` on the commit from step 2.
