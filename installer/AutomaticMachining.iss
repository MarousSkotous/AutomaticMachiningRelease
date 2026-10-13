#ifndef AM_VERSION
  #define AM_VERSION "0.0.0-dev"
#endif

#define AppName "Automatic Machining"
#define AddInDir "AutomaticMachining"

[Setup]
AppId={{DA3A4B8B-C3BD-47E5-8FD2-7555BE4F14A8}
AppName={#AppName}
AppVersion={#AM_VERSION}
AppPublisher=ccsoftcz
AppPublisherURL=https://github.com/MarousSkotous/AutomaticMachiningRelease
DefaultDirName={userappdata}\Autodesk\Autodesk Fusion 360\API\AddIns\{#AddInDir}
DisableDirPage=yes
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\build\release
OutputBaseFilename=AutomaticMachining-Windows-Setup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
Uninstallable=no
ArchitecturesAllowed=x64compatible

[InstallDelete]
Type: filesandordirs; Name: "{app}"

[Files]
Source: "..\build\release\AutomaticMachining\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Code]
function InitializeSetup(): Boolean;
begin
  Result := True;
end;
