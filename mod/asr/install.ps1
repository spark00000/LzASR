param(
    [switch]$Install,
    [switch]$Check,
    [ValidateSet('all','whisper','qwen17','qwen06')][string]$Engine = 'all'
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if ($Install -and $Check) { throw 'Choose -Install or -Check, not both' }

# External models, tools and virtual environments use a configurable runtime folder.
# No machine-wide installer, registry mutation, service change or PATH modification.
$DefaultDriveRoot = [System.IO.Path]::GetPathRoot($PSScriptRoot)
$Root = if ($env:LZASR_APP_HOME) { $env:LZASR_APP_HOME } else { Join-Path $DefaultDriveRoot 'APP\LzASR' }
$EngineRoot = Join-Path $Root 'engine'
$ModelRoot = Join-Path $Root 'models'
$Python = Join-Path $Root '.venv-qwen\Scripts\python.exe'
$WhisperExe = Join-Path $EngineRoot 'Faster-Whisper-XXL\faster-whisper-xxl.exe'
$WhisperModel = Join-Path $ModelRoot 'faster-whisper-large-v3-turbo\model.bin'
$Ffmpeg = Join-Path $EngineRoot 'ffmpeg\ffmpeg.exe'
$RepoRoot = 'https://github.com/QwenLM/Qwen3-ASR.git'

function Log([string]$Message) { Write-Output ('[LzASR] ' + $Message) }
function Require-Ok([string]$Action) {
    if ($LASTEXITCODE -ne 0) { throw ($Action + ' failed, exit=' + $LASTEXITCODE) }
}
function Model-Ready([string]$Folder) {
    if (!(Test-Path -LiteralPath (Join-Path $Folder 'config.json') -PathType Leaf)) { return $false }
    $idx = Join-Path $Folder 'model.safetensors.index.json'
    if (Test-Path -LiteralPath $idx) {
        $map = (Get-Content -LiteralPath $idx -Raw -Encoding UTF8 | ConvertFrom-Json).weight_map
        $weights = @($map.PSObject.Properties.Value | Sort-Object -Unique)
        if ($weights.Count -lt 1) { return $false }
        foreach ($name in $weights) {
            $f = Get-Item -LiteralPath (Join-Path $Folder $name) -ErrorAction SilentlyContinue
            if (!$f -or $f.Length -lt 1) { return $false }
        }
        return $true
    }
    $single = Get-Item -LiteralPath (Join-Path $Folder 'model.safetensors') -ErrorAction SilentlyContinue
    return [bool]($single -and $single.Length -gt 0)
}
function Needed([string]$Name) {
    $missing = [System.Collections.Generic.List[string]]::new()
    $needWhisper = ($Name -in @('all','whisper'))
    $need17 = ($Name -in @('all','qwen17'))
    $need06 = ($Name -in @('all','qwen06'))
    if ($needWhisper) {
        if (!(Test-Path -LiteralPath $WhisperExe)) { $missing.Add('Whisper binary') }
        if (!(Test-Path -LiteralPath $WhisperModel)) { $missing.Add('Whisper large-v3-turbo model') }
    }
    if ($need17 -or $need06) {
        if (!(Test-Path -LiteralPath $Python)) { $missing.Add('Python 3.12 / Qwen runtime') }
        if (!(Get-Command ffmpeg.exe -ErrorAction SilentlyContinue) -and !(Test-Path -LiteralPath $Ffmpeg)) {
            $missing.Add('FFmpeg')
        }
        if ($need17 -and !(Model-Ready (Join-Path $ModelRoot 'Qwen3-ASR-1.7B'))) { $missing.Add('Qwen3-ASR-1.7B') }
        if ($need06 -and !(Model-Ready (Join-Path $ModelRoot 'Qwen3-ASR-0.6B'))) { $missing.Add('Qwen3-ASR-0.6B') }
        if (!(Model-Ready (Join-Path $ModelRoot 'Qwen3-ForcedAligner-0.6B'))) { $missing.Add('Qwen3-ForcedAligner-0.6B') }
    }
    return @($missing.ToArray())
}
function Download([string]$Url, [string]$File, [long]$Size=0, [string]$Sha='') {
    if (Test-Path -LiteralPath $File) { throw ('Will not overwrite existing download: ' + $File) }
    Log ('Download: ' + $Url)
    Invoke-WebRequest -Uri $Url -OutFile $File -UseBasicParsing -TimeoutSec 1800
    $item = Get-Item -LiteralPath $File
    if ($item.Length -lt 1024 -or ($Size -gt 0 -and $item.Length -ne $Size)) {
        throw ('Download file size mismatch: ' + $File)
    }
    $digest = (Get-FileHash -LiteralPath $File -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Sha) {
        if ($digest -ne $Sha.ToLowerInvariant()) { throw ('SHA256 mismatch: ' + $File) }
        Log ('Verified SHA256: ' + $digest)
    } else {
        Log ('SHA256 local=' + $digest + '; publisher digest not available (size/TLS only)')
    }
}
function Github-Release([string]$Repository,[string]$Tag) {
    $uri = ('https://api.github.com/repos/' + $Repository + '/releases/' + $(if($Tag){'tags/'+$Tag}else{'latest'}))
    $headers = @{ 'Accept'='application/vnd.github+json'; 'User-Agent'='LzASR-Setup' }
    return Invoke-RestMethod -Uri $uri -Headers $headers -TimeoutSec 30
}
function Get-Uv {
    $local = Join-Path $EngineRoot 'uv\uv.exe'
    if (Test-Path -LiteralPath $local) { return $local }
    $global = Get-Command uv.exe -ErrorAction SilentlyContinue
    if ($global) {
        Log ('Using preinstalled uv: ' + $global.Source)
        return $global.Source
    }
    $release = Github-Release 'astral-sh/uv' ''
    $asset = @($release.assets | Where-Object { $_.name -eq 'uv-x86_64-pc-windows-msvc.zip' })
    if ($asset.Count -ne 1 -or !$asset[0].digest -or $asset[0].digest -notmatch '^sha256:[0-9a-f]{64}$') {
        throw 'Signed digest metadata missing for official uv Windows release'
    }
    $zip = Join-Path $EngineRoot 'uv-download.zip'
    Download $asset[0].browser_download_url $zip ([long]$asset[0].size) $asset[0].digest.Substring(7)
    $dest = Join-Path $EngineRoot 'uv'
    New-Item -ItemType Directory -Path $dest -Force | Out-Null
    try {
        Expand-Archive -LiteralPath $zip -DestinationPath $dest -ErrorAction Stop
        $exe = @(Get-ChildItem -LiteralPath $dest -Filter 'uv.exe' -File -Recurse)
        if ($exe.Count -ne 1) { throw 'Official uv archive does not contain one uv.exe' }
        if ($exe[0].FullName -ne $local) { Copy-Item -LiteralPath $exe[0].FullName -Destination $local }
    } finally {
        Remove-Item -LiteralPath $zip -Force -ErrorAction SilentlyContinue
    }
    return $local
}
function Ensure-Python {
    if (Test-Path -LiteralPath $Python) { return }
    $uv = Get-Uv
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $EngineRoot 'python'
    $env:UV_CACHE_DIR = Join-Path $EngineRoot 'uv-cache'
    Log ('Installing isolated Python 3.12 under ' + $Root)
    & $uv python install 3.12
    Require-Ok 'uv python install'
    & $uv venv --python 3.12 --seed (Join-Path $Root '.venv-qwen')
    Require-Ok 'uv venv'
    if (!(Test-Path -LiteralPath $Python)) { throw 'Python environment not created' }
}
function Ensure-Python-Libraries([bool]$QwenNeeded) {
    Ensure-Python
    $uv = Get-Uv
    $env:UV_CACHE_DIR = Join-Path $EngineRoot 'uv-cache'
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $EngineRoot 'python'
    $check = 'import importlib.util,sys; sys.exit(0 if all(importlib.util.find_spec(m) is not None for m in sys.argv[1:]) else 1)'
    if ($QwenNeeded) {
        & $Python -c $check 'torch' 'qwen_asr' 'huggingface_hub'
        if ($LASTEXITCODE -ne 0) {
            Log 'Installing CUDA-enabled PyTorch (cu130) and Qwen3-ASR'
            & $uv pip install --python $Python --index-url 'https://download.pytorch.org/whl/cu130' torch torchaudio
            Require-Ok 'CUDA PyTorch installation'
            $checkout = Join-Path $EngineRoot 'Qwen3-ASR'
            if (!(Test-Path -LiteralPath (Join-Path $checkout 'pyproject.toml'))) {
                if (Test-Path -LiteralPath $checkout) { throw 'Incomplete preexisting Qwen3-ASR Git checkout' }
                & git clone --depth 1 $RepoRoot $checkout
                Require-Ok 'Git clone Qwen3-ASR'
            }
            & $uv pip install --python $Python $checkout huggingface_hub
            Require-Ok 'Qwen3-ASR installation'
        }
        & $Python -c "import torch,qwen_asr;assert torch.cuda.is_available(),'CUDA not available';print('CUDA:',torch.cuda.get_device_name(0))"
        Require-Ok 'Qwen CUDA validation'
    } else {
        & $Python -c $check 'huggingface_hub'
        if ($LASTEXITCODE -ne 0) {
            & $uv pip install --python $Python huggingface_hub
            Require-Ok 'Hugging Face client installation'
        }
    }
}
function Ensure-WhisperExe {
    if (Test-Path -LiteralPath $WhisperExe) { Log 'Whisper executable present'; return }
    $release = Github-Release 'Purfview/whisper-standalone-win' 'Faster-Whisper-XXL'
    $assets = @($release.assets | Where-Object {$_.name -eq 'Faster-Whisper-XXL_r245.4_windows.7z'})
    if ($release.name -ne 'Faster-Whisper-XXL r245.4' -or $assets.Count -ne 1) {
        throw 'Official public r245.4 Windows archive not found'
    }
    $archive = Join-Path $EngineRoot 'whisper-download.7z'
    Download $assets[0].browser_download_url $archive ([long]$assets[0].size)
    $extract = Join-Path $EngineRoot 'Faster-Whisper-XXL'
    New-Item -ItemType Directory -Path $extract -Force | Out-Null
    try {
        $tar = (Get-Command tar.exe -ErrorAction Stop).Source
        & $tar -xf $archive -C $extract
        Require-Ok 'Extract Whisper 7z using Windows tar'
        $matches = @(Get-ChildItem -LiteralPath $extract -Filter faster-whisper-xxl.exe -File -Recurse)
        if ($matches.Count -ne 1) { throw 'Exactly one Whisper executable was expected' }
        $actual = $matches[0]
        # The executable hash was established from the independently functioning r245.4 installation.
        $expectedExeHash = 'CCE4C27D0D7039C93B06C6F962374EA3E8B605DBB9BF564AD23DB4C46F1A15C1'
        if ((Get-FileHash -LiteralPath $actual.FullName -Algorithm SHA256).Hash -ne $expectedExeHash) {
            throw 'Whisper extracted executable differs from verified r245.4 baseline'
        }
        if ($actual.FullName -ne $WhisperExe) {
            Get-ChildItem -LiteralPath $actual.DirectoryName -Force | ForEach-Object {
                Move-Item -LiteralPath $_.FullName -Destination $extract -ErrorAction Stop
            }
        }
        if (!(Test-Path -LiteralPath $WhisperExe)) { throw 'Whisper executable not at expected location' }
    } finally {
        Remove-Item -LiteralPath $archive -Force -ErrorAction SilentlyContinue
    }
    Log ('Whisper public r245.4 ready: ' + $WhisperExe)
}
function Download-HF([string]$Repo,[string]$Target) {
    $pythonCode = 'import sys;from huggingface_hub import snapshot_download;snapshot_download(repo_id=sys.argv[1],local_dir=sys.argv[2],max_workers=3)'
    Log ('Hugging Face snapshot: ' + $Repo + ' -> ' + $Target)
    & $Python -c $pythonCode $Repo $Target
    Require-Ok ('Download ' + $Repo)
}
function Ensure-WhisperModel {
    if (Test-Path -LiteralPath $WhisperModel) {
        $sha = (Get-FileHash -LiteralPath $WhisperModel -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($sha -ne 'e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da') {
            throw 'Whisper model SHA256 does not match published model.bin'
        }
        Log 'Whisper model SHA256 verified'
        return
    }
    Ensure-Python-Libraries $false
    Download-HF 'Infomaniak-AI/faster-whisper-large-v3-turbo' (Join-Path $ModelRoot 'faster-whisper-large-v3-turbo')
    if (!(Test-Path -LiteralPath $WhisperModel)) { throw 'Whisper model.bin missing after download' }
    Ensure-WhisperModel
}
function Ensure-Ffmpeg {
    if ((Get-Command ffmpeg.exe -ErrorAction SilentlyContinue) -or (Test-Path -LiteralPath $Ffmpeg)) { return }
    $archive = Join-Path $EngineRoot 'ffmpeg-download.zip'
    $unpack = Join-Path $EngineRoot 'ffmpeg-unpack'
    Download 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip' $archive
    New-Item -ItemType Directory -Path $unpack -Force | Out-Null
    try {
        Expand-Archive -LiteralPath $archive -DestinationPath $unpack
        $found = @(Get-ChildItem -LiteralPath $unpack -Filter ffmpeg.exe -Recurse -File)
        if ($found.Count -ne 1) { throw 'FFmpeg archive validation failed' }
        $target = Split-Path -Parent $Ffmpeg
        New-Item -ItemType Directory -Path $target -Force | Out-Null
        Copy-Item -LiteralPath $found[0].FullName -Destination $Ffmpeg -ErrorAction Stop
        if (!(Test-Path -LiteralPath $Ffmpeg)) { throw 'FFmpeg copy failed' }
    } finally {
        Remove-Item -LiteralPath $archive -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $unpack -Recurse -Force -ErrorAction SilentlyContinue
    }
}
function Ensure-QwenModel([string]$Name) {
    $path = Join-Path $ModelRoot $Name
    if (Model-Ready $path) { Log ($Name + ': present'); return }
    Download-HF ('Qwen/' + $Name) $path
    if (!(Model-Ready $path)) { throw ($Name + ' model files still incomplete') }
    Log ($Name + ': verified file layout')
}

$before = @(Needed $Engine)
Log ('Mode='+$(if($Install){'INSTALL'}else{'CHECK'})+' Engine='+$Engine+' Missing='+$before.Count)
foreach ($item in $before) { Log ('[MISSING] '+$item) }
if (!$Install) {
    if ($before.Count -gt 0) { exit 2 }
    Log '[PASS] Required files are present; no install performed'
    exit 0
}
if ($before.Count -eq 0) {
    Log '[PASS] Already installed; no download or modifications needed'
    exit 0
}
New-Item -ItemType Directory -Path $EngineRoot -Force | Out-Null
New-Item -ItemType Directory -Path $ModelRoot -Force | Out-Null
$env:HF_HOME = Join-Path $ModelRoot '.hf'
$env:HF_HUB_CACHE = Join-Path $env:HF_HOME 'hub'
$env:UV_CACHE_DIR = Join-Path $EngineRoot 'uv-cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $EngineRoot 'python'
$env:TEMP = Join-Path $EngineRoot 'temp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Path $env:TEMP -Force | Out-Null

if ($Engine -in @('all','whisper')) {
    Ensure-WhisperExe
    Ensure-WhisperModel
}
if ($Engine -in @('all','qwen17','qwen06')) {
    Ensure-Python-Libraries $true
    Ensure-Ffmpeg
    if ($Engine -in @('all','qwen17')) { Ensure-QwenModel 'Qwen3-ASR-1.7B' }
    if ($Engine -in @('all','qwen06')) { Ensure-QwenModel 'Qwen3-ASR-0.6B' }
    Ensure-QwenModel 'Qwen3-ForcedAligner-0.6B'
}
$after = @(Needed $Engine)
if ($after.Count -gt 0) { throw ('Incomplete installation: ' + ($after -join ', ')) }
Log ('[PASS] Requested engines and models installed under ' + $Root)
exit 0
