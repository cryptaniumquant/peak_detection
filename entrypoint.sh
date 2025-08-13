#!/bin/bash

# Trading Signal Bot Entrypoint Script
# This script prepares the environment and starts the bot

set -e

echo "========================================="
echo "🚀 Starting Trading Signal Bot"
echo "========================================="

# Check if local_settings.py exists
if [ ! -f "/root/app/local_settings.py" ]; then
    echo "❌ ERROR: local_settings.py not found!"
    echo "Please mount your local_settings.py file:"
    echo "  -v /path/to/your/local_settings.py:/root/app/local_settings.py"
    exit 1
fi

# Validate that we have the required files
echo "📋 Checking required files..."
required_files=(
    "run_bot.py"
    "config.py"
    "calculate_peak.py"
    "strategy_data_processor.py"
    "bot_service/run_bot.py"
    "local_settings.py"
)

for file in "${required_files[@]}"; do
    if [ ! -f "/root/app/$file" ]; then
        echo "❌ ERROR: Required file $file not found!"
        exit 1
    fi
done

echo "✅ All required files found"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p /root/app/bot_service/state
mkdir -p /root/app/visualizations
mkdir -p /root/app/processed_data

# Set permissions
chmod +x /root/app/run_bot.py

# Display configuration info (without sensitive data)
echo "⚙️  Configuration:"
echo "   - Working directory: $(pwd)"
echo "   - Python version: $(python --version)"
echo "   - Timezone: $(cat /etc/timezone)"
echo "   - Available strategies: $(ls -1 /root/app/strategy_* 2>/dev/null | wc -l) files"

# Test database connectivity (optional - comment out if not needed)
echo "🔌 Testing configuration..."
python -c "
try:
    from config import load_settings
    settings = load_settings()
    print('✅ Configuration loaded successfully')
    print(f'   - Mode: {settings.mode}')
    print(f'   - Timezone: {settings.timezone}')
    print(f'   - Real detect hours: {settings.real_detect_hours}')
except Exception as e:
    print(f'❌ Configuration error: {e}')
    exit(1)
" || exit 1

echo "========================================="
echo "🎯 Starting bot with command: $@"
echo "========================================="

# Execute the command passed to the container
exec "$@"
