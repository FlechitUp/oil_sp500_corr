# run_brent.ps1
# Activa el entorno virtual y ejecuta el script brent_ivvb11.py
# Uso:  .\run_brent.ps1

$ErrorActionPreference = "Stop"

# Ir a la carpeta donde esta este script (para que las rutas relativas funcionen)
Set-Location -Path $PSScriptRoot

$activateScript = ".\fin_env\Scripts\Activate.ps1"
$pythonScript   = "brent_ivvb11.py"

# Verificar que el entorno exista
if (-not (Test-Path $activateScript)) {
    Write-Host "ERROR: No se encontro el entorno virtual en $activateScript" -ForegroundColor Red
    Write-Host "Crealo con:  python -m venv fin_env" -ForegroundColor Yellow
    exit 1
}

# Verificar que el script de Python exista
if (-not (Test-Path $pythonScript)) {
    Write-Host "ERROR: No se encontro $pythonScript en $PSScriptRoot" -ForegroundColor Red
    exit 1
}

# Activar entorno
Write-Host "Activando entorno virtual..." -ForegroundColor Cyan
& $activateScript

# Ejecutar
Write-Host "Ejecutando $pythonScript ..." -ForegroundColor Cyan
python $pythonScript

Write-Host "Listo." -ForegroundColor Green