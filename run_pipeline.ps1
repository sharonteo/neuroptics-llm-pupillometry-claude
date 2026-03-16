# ===============================
# run_pipeline.ps1
# Full end-to-end pipeline runner
# ===============================

Write-Host "🔧 Starting full pipeline..." -ForegroundColor Cyan

# 1. Set API key for this session
$Env:ANTHROPIC_API_KEY="sk-ant-api03-..."

Write-Host "🔑 API key loaded."

# 2. Navigate to project root
Set-Location "C:\Users\sharo\projects\neuroptics-llm-pupillometry-claude"
Write-Host "📁 Moved to project directory."

# 3. Generate synthetic dataset
Write-Host "📊 Generating synthetic dataset..."
python src/data_generation.py

# 4. Train all models
Write-Host "🤖 Training models..."
python src/model.py

# 5. Launch Streamlit dashboard
Write-Host "🚀 Launching Streamlit dashboard..."
streamlit run dashboard/dashboard.py