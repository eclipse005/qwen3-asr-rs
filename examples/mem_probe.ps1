# Measure peak working set of the cpu_transcribe test process for one fixture.
# Usage: powershell -File mem_probe.ps1 <fixture> <envval: 1=default int8 | 0=f16>
param(
  [string]$fixture = "test_cpu_90s",
  [string]$envval  = "1"
)
if ($envval -eq "0") {
  $env:QASR_CPU_INT8 = "0"
  $label = "f16"
} else {
  Remove-Item Env:QASR_CPU_INT8 -ErrorAction SilentlyContinue
  $label = "int8(default)"
}
$argStr = "test --release --no-default-features --features cpu --test cpu_transcribe $fixture -- --nocapture --test-threads=1"
$p = Start-Process -FilePath cargo -ArgumentList $argStr -PassThru -WindowStyle Hidden `
     -RedirectStandardOutput "$env:TEMP\mem_$label.out" -RedirectStandardError "$env:TEMP\mem_$label.err"
$max = 0L
while (-not $p.HasExited) {
  Start-Sleep -Milliseconds 100
  Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -like "*transcribe*" } | ForEach-Object {
    if ($_.WorkingSet64 -gt $max) { $max = $_.WorkingSet64 }
  }
}
$rtfx = (Select-String -Path "$env:TEMP\mem_$label.out" -Pattern "RTFx\s+([\d.]+)x").Matches.Groups[1].Value
"{0}  fixture={1}  peakRSS={2:N0} MB  RTFx={3}x" -f $label, $fixture, ($max/1MB), $rtfx
