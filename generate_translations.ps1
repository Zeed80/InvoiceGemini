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

function Get-ToolPath([string]$toolName) {
    # Если указан явный путь к PyQt6 bin — используем его
    if ($PyQtBinPath -and (Test-Path (Join-Path $PyQtBinPath $toolName))) {
        return (Join-Path $PyQtBinPath $toolName)
    }
    # Иначе пробуем найти в PATH
    $cmd = Get-Command $toolName -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    throw "Не найден инструмент: $toolName. Укажите -PyQtBinPath или добавьте его в PATH."
}

try {
    $pylupdate = Get-ToolPath 'pylupdate6.exe'
} catch {
    # Падём обратно на pylupdate6 без .exe (если запускается из шела Python)
    $pylupdate = 'pylupdate6'
}

try {
    $lrelease = Get-ToolPath 'lrelease.exe'
} catch {
    $lrelease = 'lrelease'
}

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

# Вспомогательная функция для вызова pylupdate6 с большим числом аргументов
function Run-Pylupdate6 {
    param(
        [string[]]$Files,
        [string]$TsOut
    )
    Write-Host "📝 Генерация TS: $TsOut"
    # pylupdate6 app/**/*.py -ts translations/invoicegemini_ru.ts
    & $pylupdate @Files -ts $TsOut
    if ($LASTEXITCODE -ne 0) {
        throw "Ошибка pylupdate6 при генерации $TsOut"
    }
}

# Генерация и компиляция переводов
foreach ($lang in $Languages) {
    $tsPath = Join-Path $TranslationsDir ("invoicegemini_{0}.ts" -f $lang)
    $qmPath = Join-Path $TranslationsDir ("invoicegemini_{0}.qm" -f $lang)

    Run-Pylupdate6 -Files $sourceFiles -TsOut $tsPath

    Write-Host "🛠 Компиляция QM: $qmPath"
    & $lrelease $tsPath -qm $qmPath
    if ($LASTEXITCODE -ne 0) {
        throw "Ошибка lrelease при компиляции $tsPath"
    }
}

Write-Host "✅ Готово. Файлы .qm созданы в $TranslationsDir"
