# Downloads the latest release from GitHub and installs it for the current user (no admin needed).
$ErrorActionPreference = "Stop"
$repo = "fang3yuan/comic-translator-app"
$dest = Join-Path $env:LOCALAPPDATA "ComicTranslator"
try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Write-Host "Looking for the latest version..."
    $rel = Invoke-RestMethod "https://api.github.com/repos/$repo/releases/latest" -Headers @{ "User-Agent" = "ComicTranslatorInstaller" }
    $asset = $rel.assets | Where-Object { $_.name -eq "ComicTranslator-windows.zip" } | Select-Object -First 1
    if (-not $asset) { throw "No Windows build in the latest release yet." }
    $zip = Join-Path $env:TEMP "ComicTranslator-windows.zip"
    Write-Host "Downloading $($rel.tag_name) ($([int]($asset.size/1MB)) MB)..."
    Invoke-WebRequest $asset.browser_download_url -OutFile $zip -UseBasicParsing
    $keep = Join-Path $env:TEMP "ct_keep"; Remove-Item $keep -Recurse -Force -ErrorAction SilentlyContinue
    foreach ($d in "config", "models") { if (Test-Path "$dest\$d") { New-Item -ItemType Directory -Force $keep | Out-Null; Copy-Item "$dest\$d" $keep -Recurse -Force } }
    if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
    Write-Host "Installing to $dest ..."
    $tmp = Join-Path $env:TEMP "ct_extract"; Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
    Expand-Archive $zip -DestinationPath $tmp -Force
    Move-Item (Join-Path $tmp "ComicTranslator") $dest
    foreach ($d in "config", "models") { if (Test-Path "$keep\$d") { Copy-Item "$keep\$d" $dest -Recurse -Force } }
    $exe = Join-Path $dest "ComicTranslator.exe"
    $sh = New-Object -ComObject WScript.Shell
    foreach ($dir in [Environment]::GetFolderPath("Desktop"), (Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs")) {
        $lnk = $sh.CreateShortcut((Join-Path $dir "Comic Translator.lnk"))
        $lnk.TargetPath = $exe; $lnk.WorkingDirectory = $dest; $lnk.Save()
    }
    Remove-Item $zip, $tmp, $keep -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "`nDone! A 'Comic Translator' shortcut was added to your Desktop." -ForegroundColor Green
    Start-Process $exe
} catch {
    Write-Host "`nInstall failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "You can also download ComicTranslator-Setup.exe from: https://github.com/$repo/releases/latest"
    exit 1
}
