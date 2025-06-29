# check vcpkg
if (-not (Get-Command vcpkg -ErrorAction SilentlyContinue)) {
    Write-Error "vcpkg 未找到！请确保：`n(1) 已安装 vcpkg`n(2) 已将 vcpkg 路径添加到系统环境变量 PATH"
    exit 1
}

# install openblas and gtest
vcpkg install openblas:x64-windows gtest:x64-windows

# 3. check install status
if ($LASTEXITCODE -eq 0) {
    Write-Host "Installation succeeded!" -ForegroundColor Green
    Write-Host "openblas, gtest" -ForegroundColor Cyan
} else {
    Write-Error "Installation failed! error code: $LASTEXITCODE"
}