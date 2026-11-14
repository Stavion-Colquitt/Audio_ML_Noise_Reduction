# Call denoiser v4, fully unattended on the RTX 2060:
#   wait for the VoiceBank-DEMAND download -> unzip + resample to 16 kHz -> train (time-boxed) -> ONNX export
#   -> streaming parity check -> full 824-file test-set eval (PESQ / STOI / SI-SDR vs noisy).  Log: run_v4.log
param([double]$Hours = 8)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path; Set-Location $here
$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'      # pin by slot: bus 04 = RTX 2060 = index 0
$env:CUDA_VISIBLE_DEVICES = '0'
$log = Join-Path $here 'run_v4.log'
function Log($m) { "$(Get-Date -Format 'HH:mm:ss')  $m" | Out-File $log -Append }
"v4 run started $(Get-Date)" | Out-File $log
while (-not (Test-Path 'D:\Datasets\voicebank-demand\dl_done.txt')) { Start-Sleep 30 }
Log 'download finished; preparing 16 kHz data'
py -3.12 prep_vbd.py 2>&1 | Out-File $log -Append
py -3.12 dataset.py 2>&1 | Select-String -NotMatch 'Warning' | Out-File $log -Append
Log "training on 2060 for $Hours h"
py -3.12 train.py --hours $Hours 1> train.log 2> train.err
Log 'training finished; exporting'
py -3.12 export_onnx.py 2>&1 | Select-String 'checkpoint|saved|Error' | Out-File $log -Append
py -3.12 check_parity.py 2>&1 | Select-String 'parity|block|Error' | Out-File $log -Append
Log 'evaluating on VoiceBank-DEMAND test set (824 files)'
py -3.12 eval_vbd.py 2>&1 | Select-String -NotMatch 'Warning' | Out-File $log -Append
Log 'ALL DONE'
