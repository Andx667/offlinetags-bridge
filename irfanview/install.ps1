<#
.SYNOPSIS
    Registers offlinetags-irfanview.exe as an IrfanView external editor.

.DESCRIPTION
    Writes ExternalViewer<N> into the [Others] section of i_view64.ini, after
    saving a backup next to it. Close IrfanView first: it rewrites the INI on exit.

    The same setting can be made by hand in IrfanView under
    Options > Properties/Settings > Miscellaneous > "Set external editors".

.PARAMETER Slot
    External editor slot, 1-3. Slot N is started with Shift+N.

.PARAMETER Exe
    Path to offlinetags-irfanview.exe. Defaults to the one in this checkout's .venv.

.PARAMETER Uninstall
    Clear the slot instead of setting it.
#>
param(
    [ValidateRange(1, 3)][int]$Slot = 1,
    [string]$Exe = (Join-Path $PSScriptRoot '..\.venv\Scripts\offlinetags-irfanview.exe'),
    [string]$IniPath,
    [switch]$Uninstall
)

$ErrorActionPreference = 'Stop'

if (-not $IniPath) {
    # IrfanView keeps its settings in %APPDATA% unless the INI next to the exe says otherwise
    $IniPath = Join-Path $env:APPDATA 'IrfanView\i_view64.ini'
}
if (-not (Test-Path $IniPath)) {
    throw "IrfanView settings not found at $IniPath. Start IrfanView once, or pass -IniPath."
}
if (Get-Process -Name 'i_view64', 'i_view32' -ErrorAction SilentlyContinue) {
    throw 'IrfanView is running. Close it first, otherwise it overwrites the setting on exit.'
}

$value = ''
if (-not $Uninstall) {
    if (-not (Test-Path $Exe)) {
        throw "$Exe not found. Run 'pip install -e .' in the checkout first, or pass -Exe."
    }
    $value = (Resolve-Path $Exe).Path
}

# slot 1 has no number in the key name
$key = if ($Slot -eq 1) { 'ExternalViewer' } else { "ExternalViewer$Slot" }

Add-Type -Namespace OfflineTags -Name Ini -MemberDefinition @'
[DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
public static extern bool WritePrivateProfileString(string section, string key, string value, string path);
[DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
public static extern uint GetPrivateProfileString(string section, string key, string def, System.Text.StringBuilder buffer, uint size, string path);
'@

$current = New-Object System.Text.StringBuilder 1024
[void][OfflineTags.Ini]::GetPrivateProfileString('Others', $key, '', $current, 1024, $IniPath)
if ($current.Length -gt 0 -and $current.ToString() -ne $value -and -not $Uninstall) {
    throw "Slot $Slot is already used by '$current'. Choose another with -Slot."
}

$backup = "$IniPath.offlinetags.bak"
Copy-Item $IniPath $backup -Force

# the Win32 call keeps the file's UTF-16 encoding intact
# a null value removes the key
$newValue = if ($Uninstall) { [NullString]::Value } else { $value }
if (-not [OfflineTags.Ini]::WritePrivateProfileString('Others', $key, $newValue, $IniPath)) {
    throw "Could not write $IniPath (Win32 error $([Runtime.InteropServices.Marshal]::GetLastWin32Error()))."
}

if ($Uninstall) {
    Write-Host "Cleared external editor slot $Slot."
} else {
    Write-Host "Registered $value as external editor $Slot. In IrfanView, press Shift+$Slot on an open photo."
}
Write-Host "Backup of the previous settings: $backup"
