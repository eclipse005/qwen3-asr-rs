# Measure peak working set of the cpu_transcribe test process for one fixture.
# Usage: powershell -File mem_probe.ps1 <fixture>
# Note: in-process peak RSS is now reported directly by cpu_transcribe.rs (Win32
# GetProcessMemoryInfo); this script is just a fallback external poller. INT8 is permanent.
param(
  [string]$fixture = "test_cpu_90s"
)
$argStr = "test --release --no-default-features --features cpu --test cpu_transcribe $fixture -- --nocapture --test-threads=1"
$p = Start-Process -FilePath cargo -ArgumentList $argStr -PassThru -WindowStyle Hidden `
     -RedirectStandardOutput "$env:TEMP\mem_$fixture.out" -RedirectStandardError "$env:TEMP\mem_$fixture.err"
$max = 0L
while (-not $p.HasExited) {
  Start-Sleep -Milliseconds 100
  Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -like "*transcribe*" } | ForEach-Object {
    if ($_.WorkingSet64 -gt $max) { $max = $_.WorkingSet64 }
  }
}
$rtfx = (Select-String -Path "$env:TEMP\mem_$fixture.out" -Pattern "RTFx\s+([\d.]+)x").Matches.Groups[1].Value
"fixture={0}  peakRSS={1:N0} MB  RTFx={2}x" -f $fixture, ($max/1MB), $rtfx
