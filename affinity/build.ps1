<#
.SYNOPSIS
    Builds affinity\build\OfflineTags.8bf (64-bit) with the Visual Studio C++ tools.

.DESCRIPTION
    Needs Visual Studio or the Build Tools with the "Desktop development with C++"
    workload. The Adobe Photoshop SDK is not required.
#>
$ErrorActionPreference = 'Stop'

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path $vswhere)) {
    throw 'Visual Studio installer not found. Install Visual Studio or the Build Tools with the C++ workload.'
}
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) {
    throw 'No Visual Studio installation with the C++ x64 tools found.'
}
$vcvars = Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat'
# vcvars64.bat calls vswhere itself and expects it on the PATH
$env:PATH += ";$(Split-Path $vswhere)"

$out = Join-Path $PSScriptRoot 'build'
New-Item -ItemType Directory -Force $out | Out-Null

# /MT links the C runtime statically, so the plug-in has no redistributable to install
$steps = @(
    "`"$vcvars`" >nul"
    "cd /d `"$out`""
    'rc /nologo /fo pipl.res ..\pipl.rc'
    'cl /nologo /std:c11 /O2 /W4 /WX /MT /LD ..\plugin.c pipl.res /Fe:OfflineTags.8bf /link /NOLOGO user32.lib'
) -join ' && '
cmd /c $steps
if ($LASTEXITCODE -ne 0) {
    throw "Build failed with exit code $LASTEXITCODE."
}
Write-Host "Built $(Join-Path $out 'OfflineTags.8bf')"
