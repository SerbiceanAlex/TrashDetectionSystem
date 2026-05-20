$ErrorActionPreference = "Stop"

$ThesisRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$MiKTeXPath = Join-Path $env:LOCALAPPDATA "Programs\MiKTeX\miktex\bin\x64"

function Resolve-Tool($Name) {
    $fromPath = Get-Command $Name -ErrorAction SilentlyContinue
    if ($fromPath) {
        return $fromPath.Source
    }

    $fromMiKTeX = Join-Path $MiKTeXPath $Name
    if (Test-Path -LiteralPath $fromMiKTeX) {
        return $fromMiKTeX
    }

    throw "Nu am găsit $Name. Verifică instalarea MiKTeX sau repornește terminalul."
}

$XeLaTeX = Resolve-Tool "xelatex.exe"
$BibTeX = Resolve-Tool "bibtex.exe"

Push-Location $ThesisRoot
try {
    & $XeLaTeX -interaction=nonstopmode -synctex=1 main.tex
    & $BibTeX main
    & $XeLaTeX -interaction=nonstopmode -synctex=1 main.tex
    & $XeLaTeX -interaction=nonstopmode -synctex=1 main.tex
    Write-Host "PDF generat: $ThesisRoot\main.pdf"
}
finally {
    Pop-Location
}
