# Continue an interrupted v4 run from checkpoints\latest.pth on the RTX 2060 (same total length, same MLflow run),
# then export + parity + test-set eval. Used by the overnight check-ins if the main run dies.
$here = Split-Path -Parent $MyInvocation.MyCommand.Path; Set-Location $here
$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'      # bus 04 = RTX 2060 = index 0
$env:CUDA_VISIBLE_DEVICES = '0'
$log = Join-Path $here 'run_v4.log'
function Log($m) { "$(Get-Date -Format 'HH:mm:ss')  $m" | Out-File $log -Append }
Log 'RESUMING v4 from latest.pth'
if (Test-Path train.log) { Move-Item train.log "train_$(Get-Date -Format HHmm).log.old" -Force }
py -3.12 train.py --resume --hours 8 1> train.log 2> train.err
Log 'training finished; exporting'
py -3.12 export_onnx.py 2>&1 | Select-String 'checkpoint|saved|Error' | Out-File $log -Append
py -3.12 check_parity.py 2>&1 | Select-String 'parity|block|Error' | Out-File $log -Append
Log 'evaluating on VoiceBank-DEMAND test set (824 files)'
py -3.12 eval_vbd.py 2>&1 | Select-String -NotMatch 'Warning' | Out-File $log -Append
Log 'ALL DONE'
