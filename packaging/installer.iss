; Inno Setup 安装器脚本
;
; 只装主程序（几百 MB）。TTS 推理运行时约 9GB，做成首次启动按需下载，
; 所以这里**不包含** gsv\ 的内容，只建一个空目录并留一份说明。
;
; 编译：
;   ISCC.exe packaging\installer.iss
; 或者：
;   python tools\build_installer.py --inno
;
; 依赖：先跑 tools\build_installer.py 生成 build\installer_tree\

#define MyAppName "自动读屏台"
#define MyAppNameEn "Auto Screen Reader"
#define MyAppVersion "0.8.3"
#define MyAppPublisher "screenshot2Speach"
#define TreeDir "..\build\installer_tree"
#define OutDir  "..\build\installer"

[Setup]
AppId={{8E3F1C42-6A5D-4B77-9E21-3C0A5B7D4F18}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppNameEn}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir={#OutDir}
OutputBaseFilename=AutoScreenReader-Setup-{#MyAppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayName={#MyAppName}

; ============================================================
;  关键：按用户安装，**不请求管理员权限**
;
;  这样双击安装包时**完全不会弹出 UAC 的「未知发布者」对话框**。
;  装到 %LOCALAPPDATA%\AutoScreenReader，卸载项也在当前用户下。
;
;  代价：装不到 Program Files（但我们的 TTS 运行时本来就有 9GB，
;  装在用户目录反而更合理，也避免了 Program Files 写不进去的问题）。
;
;  PrivilegesRequiredOverridesAllowed=dialog 让有需要的人自己选「为所有用户安装」，
;  那时才会弹 UAC —— 但默认路径不会。
;
;  如果以后买了代码签名证书，把 SignTool 和 SignedUninstaller 配上即可，
;  UAC 里就会显示你的名字而不是「未知发布者」。
; ============================================================
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

[Languages]
Name: "chinese"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加任务:"

[Files]
; 整个安装目录原样搬过去（已经由 build_installer.py 组装好）
Source: "{#TreeDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Dirs]
; 空目录不会有文件，要显式创建：声库目录与运行时目录
Name: "{app}\voices"; Permissions: users-modify
Name: "{app}\gsv";    Permissions: users-modify
Name: "{app}\data";   Permissions: users-modify

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\读屏台.exe"
Name: "{autoprograms}\{#MyAppName}（带控制台排障）"; Filename: "{app}\run_console.bat"
Name: "{autoprograms}\使用说明"; Filename: "{app}\README.txt"
Name: "{autoprograms}\卸载 {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\读屏台.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\README.txt"; Description: "查看使用说明"; Flags: postinstall shellexec nowait skipifsilent
Filename: "{app}\读屏台.exe"; Description: "立即启动"; Flags: postinstall nowait skipifsilent unchecked

[UninstallDelete]
; 让用户知道这些是程序生成的，卸载时一并清掉
Type: filesandordirs; Name: "{app}\data\tts_cache"
