#requires -Version 5.1
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$paths = @(
  (Join-Path $root "data\bdd100k\raw\labels_train.json"),
  (Join-Path $root "data\bdd100k\raw\labels_val.json"),
  (Join-Path $root "data\bdd100k\images\train"),
  (Join-Path $root "data\bdd100k\images\val")
)
foreach ($path in $paths) {
    if (Test-Path $path) { Write-Host "OK: $path" }
    else { Write-Warning "MISSING: $path" }
}
Write-Host "Expected by RUN_PILOT.cmd and RUN_ALL.cmd; no files modified."
