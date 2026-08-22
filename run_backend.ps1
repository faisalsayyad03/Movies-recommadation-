param(
    [int]$Port = 8000
)

if ($env:PORT) {
    $Port = [int]$env:PORT
}

python -m uvicorn main:app --host 0.0.0.0 --port $Port