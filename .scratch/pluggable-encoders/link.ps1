$ErrorActionPreference = 'Continue'
$root = $PSScriptRoot

function Read-Utf8 {
    param([string]$Path)
    return [IO.File]::ReadAllText($Path, [Text.Encoding]::UTF8)
}

function Invoke-Gh {
    param([string[]]$GhArgs, [int]$Attempts = 4)
    for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
        $output = & gh @GhArgs 2>&1
        $code = $LASTEXITCODE
        if ($code -eq 0) { return @($output | ForEach-Object { "$_" }) }
        if ($attempt -eq $Attempts) {
            $text = ($output | ForEach-Object { "$_" }) -join ' '
            throw ("gh " + ($GhArgs -join ' ') + " failed ($code): " + $text)
        }
        Start-Sleep -Seconds (4 * $attempt)
    }
}

function Write-BodyFile {
    param([string]$Text)
    $tmp = [IO.Path]::GetTempFileName() + '.md'
    [IO.File]::WriteAllText($tmp, $Text, (New-Object Text.UTF8Encoding($false)))
    return $tmp
}

$repo = (& gh repo view --json nameWithOwner --jq .nameWithOwner)
Write-Host "REPO: $repo"
$specNumber = 1
$numbers = @{ T1 = 2; T2 = 3; T3 = 4; T4 = 5; T5 = 6; T6 = 7; T7 = 8 }

$defs = @(
    @{ key = 'T1'; file = '01-models-package-move.md'; blockedBy = @() },
    @{ key = 'T2'; file = '02-model-artifact-loading.md'; blockedBy = @('T1') },
    @{ key = 'T3'; file = '03-two-axis-api-and-default-mlp-encoder.md'; blockedBy = @('T2') },
    @{ key = 'T4'; file = '04-dcn-v2-encoder.md'; blockedBy = @('T3') },
    @{ key = 'T5'; file = '05-senet-encoder.md'; blockedBy = @('T3') },
    @{ key = 'T6'; file = '06-baseline-v2-comparison.md'; blockedBy = @('T5') },
    @{ key = 'T7'; file = '07-permutation-importance-rerun.md'; blockedBy = @('T3') }
)

$editFailures = 0
foreach ($d in $defs) {
    $text = (Read-Utf8 (Join-Path $root ("tickets/" + $d.file))).Replace('#{{PARENT}}', "#$specNumber")
    foreach ($key in $d.blockedBy) { $text = $text.Replace("#{{$key}}", "#" + $numbers[$key]) }
    $tmp = Write-BodyFile $text
    try {
        Invoke-Gh @('issue', 'edit', "$($numbers[$d.key])", '--body-file', $tmp) | Out-Null
        Write-Host ("BODY_OK {0} #{1}" -f $d.key, $numbers[$d.key])
    } catch {
        $editFailures++
        Write-Host ("BODY_FAILED {0}: {1}" -f $d.key, $_)
    } finally {
        Remove-Item -LiteralPath $tmp -Force
    }
    Start-Sleep -Seconds 2
}

$edgeFailures = 0
$subIssueFailures = 0
foreach ($d in $defs) {
    $child = $numbers[$d.key]
    $childId = (& gh api "repos/$repo/issues/$child" --jq .id)
    try {
        Invoke-Gh @('api', '--method', 'POST', "repos/$repo/issues/$specNumber/sub_issues", '-F', "sub_issue_id=$childId") | Out-Null
        Write-Host ("SUB_ISSUE_OK {0}" -f $d.key)
    } catch {
        $subIssueFailures++
        Write-Host ("SUB_ISSUE_FAILED {0}: {1}" -f $d.key, $_)
    }
    Start-Sleep -Seconds 2
    foreach ($key in $d.blockedBy) {
        $blockerId = (& gh api "repos/$repo/issues/$($numbers[$key])" --jq .id)
        try {
            Invoke-Gh @('api', '--method', 'POST', "repos/$repo/issues/$child/dependencies/blocked_by", '-F', "issue_id=$blockerId") | Out-Null
            Write-Host ("EDGE_OK {0} blocked_by {1}" -f $d.key, $key)
        } catch {
            $edgeFailures++
            Write-Host ("EDGE_FAILED {0} blocked_by {1}: {2}" -f $d.key, $key, $_)
        }
        Start-Sleep -Seconds 2
    }
}

Write-Host ("SUMMARY spec=#{0} body_failed={1} edges_failed={2} sub_issue_failed={3}" -f $specNumber, $editFailures, $edgeFailures, $subIssueFailures)
$ordered = [ordered]@{ spec = $specNumber }
foreach ($d in $defs) { $ordered[$d.key] = $numbers[$d.key] }
$ordered['bodyFailures'] = $editFailures
$ordered['edgeFailures'] = $edgeFailures
$ordered['subIssueFailures'] = $subIssueFailures
$ordered | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $root 'published.json') -Encoding UTF8
Write-Host 'WROTE published.json'
