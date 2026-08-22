param(
    [int]$Port = 8501
)

if ($env:PORT) {
    $Port = [int]$env:PORT
}

python -m streamlit run app.py --server.address 0.0.0.0 --server.port $Port