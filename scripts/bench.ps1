#!/usr/bin/env pwsh
# Run each transcribe_burn test as a SEPARATE `cargo test` invocation so the
# CUDA driver / cuBLAS workspace / kernel cache start cold for every test.
# This avoids cross-test cache-warmup artifacts that confound RTFx.
#
# Usage:  pwsh scripts/bench.ps1              # run all 0.6B tests (default)
#         pwsh scripts/bench.ps1 q06_15s      # run one specific test
#         pwsh scripts/bench.ps1 q17           # run all 1.7B tests
#
# RTFx reference (HANDOFF.md §0, P104-100, 0.6B f16, single-test):
#   15s en   24.00x   30s zh   20.70x   90s en   18.87x
#   180s zh  17.34x   180s en  17.34x   89s ja   20.45x

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

# Test list — short_name | test_fn
$tests = @(
    @{ name = 'q06_sample1';  fn = 'test_q06_sample1'  }
    @{ name = 'q06_15s';      fn = 'test_q06_15s'      }
    @{ name = 'q06_30s';      fn = 'test_q06_30s'      }
    @{ name = 'q06_89s_ja';   fn = 'test_q06_89s_ja'   }
    @{ name = 'q06_90s';      fn = 'test_q06_90s'      }
    @{ name = 'q06_180s';     fn = 'test_q06_180s'     }
    @{ name = 'q06_180s_en';  fn = 'test_q06_180s_en'  }
    @{ name = 'q17_15s';      fn = 'test_q17_15s'      }
    @{ name = 'q17_30s';      fn = 'test_q17_30s'      }
    @{ name = 'q17_89s_ja';   fn = 'test_q17_89s_ja'   }
)

# Filter
$filter = $args
if ($filter.Count -gt 0) {
    $pat = ($filter -join '|')
    $tests = $tests | Where-Object { $_.name -match $pat -or $_.fn -match $pat }
    if ($tests.Count -eq 0) {
        Write-Error "No tests match pattern: $($filter -join ' ')"
    }
}

# RTFx table (from HANDOFF.md §0)
$refTable = @{
    'q06_sample1'  = $null   # not in HANDOFF
    'q06_15s'      = 24.00
    'q06_30s'      = 20.70
    'q06_89s_ja'   = 20.45
    'q06_90s'      = 18.87
    'q06_180s'     = 17.34
    'q06_180s_en'  = 17.34
    'q17_15s'      = 11.37
    'q17_30s'      = 10.05
    'q17_89s_ja'   = $null   # not in HANDOFF
}

$results = @()

foreach ($t in $tests) {
    $name = $t.name
    $fn   = $t.fn
    Write-Host ""
    Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  $name  ($fn)" -ForegroundColor Cyan
    Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Cyan

    # One cargo invocation per test — fresh CUDA context, fresh cache.
    # Tee the output: print to console AND capture to a temp file so we can
    # post-process it (cargo exits before the test's stdout flushes under
    # --nocapture, so we can't rely on the captured pipeline alone).
    $tmp = New-TemporaryFile
    cargo test --release --features cuda --test transcribe_burn $fn -- --ignored --nocapture *> $tmp.FullName
    $cargoExit = $LASTEXITCODE
    $out = Get-Content $tmp.FullName -Raw
    Remove-Item $tmp.FullName -Force

    # Surface the test's own status line ("0.6B-15s | 0.622s elapsed | RTFx 24.10x")
    $rtfx = $null
    if ($out -match 'RTFx\s+([\d.]+)x') {
        $rtfx = [double]$Matches[1]
    }

    $results += [pscustomobject]@{
        Test   = $name
        RTFx   = $rtfx
        Ref    = $refTable[$name]
    }

    if ($cargoExit -ne 0) {
        Write-Host "  ✗ FAILED (cargo exit $cargoExit)" -ForegroundColor Red
    } else {
        Write-Host "  ✓ $fn  (RTFx = $rtfx x)" -ForegroundColor Green
    }
}

# Summary
Write-Host ""
Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Summary  (each row = one cold-start cargo test invocation)" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Cyan
$results | Format-Table -AutoSize

# Regression check
$regressions = $results | Where-Object { $_.Ref -and $_.RTFx -and ($_.RTFx -lt $_.Ref * 0.95) }
if ($regressions) {
    Write-Host "  ⚠ REGRESSION (>5% below HANDOFF ref):" -ForegroundColor Yellow
    $regressions | ForEach-Object {
        $delta = ($_.RTFx / $_.Ref - 1) * 100
        Write-Host "    $($_.Test): $($_.RTFx)x vs ref $($_.Ref)x  ($([Math]::Round($delta,1))%)" -ForegroundColor Yellow
    }
    exit 1
} else {
    Write-Host "  ✓ All measured RTFx within 5% of HANDOFF ref" -ForegroundColor Green
    exit 0
}
