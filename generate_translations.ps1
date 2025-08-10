#Requires -Version 5.1
param(
    [string]$PyQtBinPath = "",   # Путь к каталогу, где лежат pylupdate6.exe и lrelease.exe (если не в PATH)
    [string[]]$Languages = @('ru','en')  # Коды языков (совпадают с invoicegemini_<lang>.qm)
)

# Определяем корень проекта и каталоги
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ProjectRoot
$TranslationsDir = Join-Path $ProjectRoot 'translations'

if (-not (Test-Path $TranslationsDir)) {
    New-Item -ItemType Directory -Path $TranslationsDir | Out-Null
}

function Resolve-Tool {
    param(
        [string[]]$Candidates
    )
    foreach ($name in $Candidates) {
        # 1) Явный путь
        if ($PyQtBinPath) {
            $full = Join-Path $PyQtBinPath $name
            if (Test-Path $full) { return $full }
        }
        # 2) В PATH
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd) { return $cmd.Source }
    }
    return $null
}

# Списки возможных имён инструментов (Qt/PyQt/PySide)
$pylupdateCandidates = @('pylupdate6.exe','pylupdate6','pyside6-lupdate.exe','pyside6-lupdate','lupdate.exe','lupdate')
$lreleaseCandidates  = @('lrelease.exe','lrelease','pyside6-lrelease.exe','pyside6-lrelease')

$pylupdate = Resolve-Tool -Candidates $pylupdateCandidates
$lrelease  = Resolve-Tool -Candidates $lreleaseCandidates

if (-not $pylupdate) { throw "Не найден инструмент pylupdate/lupdate. Укажите -PyQtBinPath или добавьте в PATH (проверьте Qt/PySide6)." }
if (-not $lrelease)  { throw "Не найден инструмент lrelease. Укажите -PyQtBinPath или добавьте в PATH (проверьте Qt/PySide6)." }

Write-Host "📂 Корень проекта: $ProjectRoot"
Write-Host "🌐 Каталог переводов: $TranslationsDir"
Write-Host "🛠 pylupdate: $pylupdate"
Write-Host "🛠 lrelease:  $lrelease"

# Собираем список исходников .py
$sourceFiles = Get-ChildItem -Recurse -Include *.py -Path (Join-Path $ProjectRoot 'app'), (Join-Path $ProjectRoot 'debug_runner.py') | ForEach-Object { $_.FullName }
if (-not $sourceFiles -or $sourceFiles.Count -eq 0) {
    Write-Error "Не найдены исходники .py для извлечения строк."
    exit 1
}

function Run-Pylupdate6 {
    param(
        [string[]]$Files,
        [string]$TsOut
    )
    Write-Host "📝 Генерация TS: $TsOut"
    & $pylupdate @Files -ts $TsOut | Out-String | Write-Host
    if ($LASTEXITCODE -ne 0) { throw "Ошибка pylupdate/lupdate при генерации $TsOut" }
}

foreach ($lang in $Languages) {
    $tsPath = Join-Path $TranslationsDir ("invoicegemini_{0}.ts" -f $lang)
    $qmPath = Join-Path $TranslationsDir ("invoicegemini_{0}.qm" -f $lang)

    Run-Pylupdate6 -Files $sourceFiles -TsOut $tsPath

    Write-Host "🛠 Компиляция QM: $qmPath"
    & $lrelease $tsPath -qm $qmPath | Out-String | Write-Host
    if ($LASTEXITCODE -ne 0) { throw "Ошибка lrelease при компиляции $tsPath" }
}

Write-Host "✅ Готово. Файлы .qm созданы в $TranslationsDir"
