# Windows launcher for runner/augment_coupled.py, with the same Python setup as run.ps1.
# Every argument is forwarded, e.g. --source RUN_DIRECTORY [--plan] [--output COPY].
$ErrorActionPreference = 'Stop'
$benchmarkPython = Join-Path $env:USERPROFILE 'anaconda3/python.exe'
if (-not (Test-Path -LiteralPath $benchmarkPython)) {
    $benchmarkPython = (Get-Command python -ErrorAction Stop).Source
}
$savedPythonPath = $env:PYTHONPATH
try {
    $poolVersion = & $benchmarkPython -c 'import threadpoolctl; print(threadpoolctl.__version__)'
    if ($LASTEXITCODE -ne 0) { throw 'Install benchmarks/runner/requirements-publication.txt in your Python environment.' }
    if ([version]$poolVersion -lt [version]'3.6.0') {
        # The compatible package is already present in the user's Conda cache.
        $cachedPool = Join-Path $env:USERPROFILE 'anaconda3/pkgs/threadpoolctl-3.6.0-pyhecae5ae_0/site-packages'
        if (-not (Test-Path -LiteralPath (Join-Path $cachedPool 'threadpoolctl.py'))) {
            throw 'threadpoolctl >= 3.6 is required. Install benchmarks/runner/requirements-publication.txt.'
        }
        $env:PYTHONPATH = if ($savedPythonPath) { "$cachedPool;$savedPythonPath" } else { $cachedPool }
        Write-Host 'Using cached threadpoolctl 3.6.0 for this process; the Python installation is unchanged.'
    }
    & $benchmarkPython (Join-Path $PSScriptRoot '..\runner\augment_coupled.py') @args
    $benchmarkExitCode = $LASTEXITCODE
} finally {
    $env:PYTHONPATH = $savedPythonPath
}
exit $benchmarkExitCode
