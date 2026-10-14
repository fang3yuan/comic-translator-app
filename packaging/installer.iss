; Inno Setup script - builds ComicTranslator-Setup.exe (run by GitHub Actions)
#define AppName "Comic Translator"
#ifndef AppVersion
  #define AppVersion "2.0.0"
#endif

[Setup]
AppId={{6F1D2C3A-6B1E-4E55-9C0B-0C0A1C7A5A01}
AppName={#AppName}
AppVersion={#AppVersion}
DefaultDirName={localappdata}\ComicTranslator
DefaultGroupName={#AppName}
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename=ComicTranslator-Setup
SetupIconFile=..\assets\icon.ico
UninstallDisplayIcon={app}\ComicTranslator.exe
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern
DisableProgramGroupPage=yes

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: checkedonce

[Files]
Source: "..\dist\ComicTranslator\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\ComicTranslator.exe"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\ComicTranslator.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\ComicTranslator.exe"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent
