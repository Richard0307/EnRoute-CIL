param(
    [Parameter(Position = 0)]
    [string]$Message,

    [string]$Remote = "origin",

    [string]$Branch
)

$ErrorActionPreference = "Stop"

function Fail([string]$Text) {
    Write-Host "ERROR: $Text" -ForegroundColor Red
    exit 1
}

try {
    git rev-parse --is-inside-work-tree *> $null
    if ($LASTEXITCODE -ne 0) {
        Fail "Current directory is not a Git repository."
    }

    if ([string]::IsNullOrWhiteSpace($Branch)) {
        $Branch = (git rev-parse --abbrev-ref HEAD).Trim()
    }

    if ([string]::IsNullOrWhiteSpace($Message)) {
        $Message = Read-Host "Enter commit message"
    }

    if ([string]::IsNullOrWhiteSpace($Message)) {
        Fail "Commit message cannot be empty."
    }

    Write-Host "Staging changes..." -ForegroundColor Cyan
    git add -A

    git diff --cached --quiet
    if ($LASTEXITCODE -eq 0) {
        Write-Host "No staged changes to commit. Nothing to push." -ForegroundColor Yellow
        exit 0
    }

    Write-Host "Creating commit..." -ForegroundColor Cyan
    git commit -m $Message

    Write-Host "Pushing to $Remote/$Branch ..." -ForegroundColor Cyan
    git push $Remote $Branch

    Write-Host "Done: pushed to $Remote/$Branch" -ForegroundColor Green
}
catch {
    Fail $_.Exception.Message
}
