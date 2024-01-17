$env:LOCAL_LLM_API_KEY = Get-Content -Path ".env" | Select-String -Pattern "LOCAL_LLM_API_KEY" | ForEach-Object { $_.ToString().Split("=")[1] }
if ($null -eq $env:LOCAL_LLM_API_KEY) {
    $env:LOCAL_LLM_API_KEY = ""
}
$env:THREADS = Get-Content -Path ".env" | Select-String -Pattern "THREADS" | ForEach-Object { $_.ToString().Split("=")[1] }
if ($null -eq $env:THREADS) {
    $env:THREADS = [Environment]::ProcessorCount - 1
}
$env:MAIN_GPU = Get-Content -Path ".env" | Select-String -Pattern "MAIN_GPU" | ForEach-Object { $_.ToString().Split("=")[1] }
if ($null -eq $env:MAIN_GPU) {
    $env:MAIN_GPU = "0"
}
$env:GPU_LAYERS = Get-Content -Path ".env" | Select-String -Pattern "GPU_LAYERS" | ForEach-Object { $_.ToString().Split("=")[1] }
if ($null -eq $env:GPU_LAYERS) {
    $env:GPU_LAYERS = "0"
}
$env:CMAKE_ARGS = Get-Content -Path ".env" | Select-String -Pattern "CMAKE_ARGS" | ForEach-Object { $_.ToString().Split("=")[1] }
if ($null -eq $env:CMAKE_ARGS) {
    $env:CMAKE_ARGS = ""
}
Write-Host $env:CMAKE_ARGS
$env:CUDA_DOCKER_ARCH = Get-Content -Path ".env" | Select-String -Pattern "CUDA_DOCKER_ARCH" | ForEach-Object { $_.ToString().Split("=")[1] }
if ($null -eq $env:CUDA_DOCKER_ARCH) {
    $env:CUDA_DOCKER_ARCH = ""
}
if ($env:GPU_LAYERS -ne "0") {
    $env:CUDA_DOCKER_ARCH = "all"
    if ($env:CMAKE_ARGS -ne "-DLLAMA_CUBLAS=on") {
        # if length of $env:CMAKE_ARGS is 0
        if ($env:CMAKE_ARGS.Length -eq 0) {
            write-host "Installing llama-cpp-python with cublas support"
            $env:CMAKE_ARGS = "-DLLAMA_CUBLAS=on"
            Add-Content -Path ".env" -Value "CMAKE_ARGS=$env:CMAKE_ARGS"
            & pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
        }
    }
}
& uvicorn app:app --host 0.0.0.0 --port 8091 --workers 4 --proxy-headers
