function Test-NvidiaGpuPresent {
    if ($IsWindows) { # Windows-specific check
        $nvidiaPresent = Get-WmiObject Win32_VideoController | Where-Object { $_.Name -like "*NVIDIA*" }
    } elseif ($IsLinux) { # Linux-specific check
        $nvidiaPresent = lspci | Select-String "NVIDIA"
    } else {
        Write-Error "Unsupported operating system"
        return $false
    }
    return $null -ne $nvidiaPresent
}

$env:RUN_WITHOUT_DOCKER = Get-Content -Path ".env" | Select-String -Pattern "RUN_WITHOUT_DOCKER" | ForEach-Object { $_.ToString().Split("=")[1] }
if ($null -eq $env:RUN_WITHOUT_DOCKER) {
    $env:RUN_WITHOUT_DOCKER = ""
}
$env:EZLOCALAI_API_KEY = Get-Content -Path ".env" | Select-String -Pattern "EZLOCALAI_API_KEY" | ForEach-Object { $_.ToString().Split("=")[1] }
if ($null -eq $env:EZLOCALAI_API_KEY) {
    $env:EZLOCALAI_API_KEY = ""
}
$env:THREADS = Get-Content -Path ".env" | Select-String -Pattern "THREADS" | ForEach-Object { $_.ToString().Split("=")[1] }
if ($null -eq $env:THREADS) {
    $env:THREADS = [Environment]::ProcessorCount - 2
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
if ($env:CMAKE_ARGS -eq "-DLLAMA_CUBLAS") {
    $env:CMAKE_ARGS = "-DLLAMA_CUBLAS=on"
}
$env:CUDA_DOCKER_ARCH = Get-Content -Path ".env" | Select-String -Pattern "CUDA_DOCKER_ARCH" | ForEach-Object { $_.ToString().Split("=")[1] }
if ($null -eq $env:CUDA_DOCKER_ARCH) {
    $env:CUDA_DOCKER_ARCH = ""
}
$env:AUTO_UPDATE = Get-Content -Path ".env" | Select-String -Pattern "AUTO_UPDATE" | ForEach-Object { $_.ToString().Split("=")[1] }
if ($null -eq $env:AUTO_UPDATE) {
    $env:AUTO_UPDATE = "true"
}
$env:USE_VULKAN = Get-Content -Path ".env" | Select-String -Pattern "USE_VULKAN" | ForEach-Object { $_.ToString().Split("=")[1] }
if ($null -eq $env:USE_VULKAN) {
    $env:USE_VULKAN = "false"
}
if (Test-NvidiaGpuPresent) {
    if($env:USE_VULKAN -eq "true") {
        Write-Host "NVIDIA GPU detected, using Vulkan image.."
    } else {
        Write-Host "NVIDIA GPU detected, using CUDA image.."
    }
    $env:CUDA_DOCKER_ARCH = "all"
    if ($env:CMAKE_ARGS -ne "-DLLAMA_CUBLAS=on") {
        # if length of $env:CMAKE_ARGS is 0
        if ($env:CMAKE_ARGS.Length -eq 0) {
            $env:CMAKE_ARGS = "-DLLAMA_CUBLAS=on"
            if( $env:RUN_WITHOUT_DOCKER.Length -ne 0) {
                Add-Content -Path ".env" -Value "CMAKE_ARGS=$env:CMAKE_ARGS"
                & pip install llama-cpp-python --upgrade --force-reinstall
            }
        }
    }
} else {
    Write-Host "No NVIDIA GPU detected, using CPU image.."
}

if( $env:RUN_WITHOUT_DOCKER.Length -ne 0) {
    if ($env:AUTO_UPDATE -eq "true") {
        git pull
    }
    & uvicorn app:app --host 0.0.0.0 --port 8091 --workers 1 --proxy-headers
} else {
    if ($env:CUDA_DOCKER_ARCH.Length -ne 0) {
        if($env:USE_VULKAN -eq "true") {
            docker-compose -f docker-compose-vulkan.yml down
            if ($env:AUTO_UPDATE -eq "true") {
                Write-Host "Build latest images, please wait.."
                docker-compose -f docker-compose-vulkan.yml build
            }
            Write-Host "Starting server, please wait.."
            docker-compose -f docker-compose-vulkan.yml up
        } else {
            docker-compose -f docker-compose-cuda.yml down
            if ($env:AUTO_UPDATE -eq "true") {
                Write-Host "Build latest images, please wait.."
                docker-compose -f docker-compose-cuda.yml build
            }
            Write-Host "Starting server, please wait.."
            docker-compose -f docker-compose-cuda.yml up
        }
    } else {
        docker-compose down
        if ($env:AUTO_UPDATE -eq "true") {
            Write-Host "Pulling latest images, please wait.."
            docker-compose pull
        }
        Write-Host "Starting server, please wait.."
        docker-compose up
    }
}