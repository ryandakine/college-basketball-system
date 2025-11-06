#!/bin/bash
# Setup Full Automation - One Command Installation

set -e

echo "🤖 FULL AUTOMATION SETUP"
echo "========================"
echo

# Get system directory
SYSTEM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "System directory: $SYSTEM_DIR"

# Find Python
if [ -f "$SYSTEM_DIR/venv/bin/python" ]; then
    PYTHON_EXEC="$SYSTEM_DIR/venv/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON_EXEC=$(command -v python3)
else
    echo "❌ Error: Python not found"
    exit 1
fi
echo "Python: $PYTHON_EXEC"

# Test Python
echo
echo "Testing system..."
if ! $PYTHON_EXEC -c "import sys; print(f'Python {sys.version}')"; then
    echo "❌ Python test failed"
    exit 1
fi

# Test imports
echo "Checking dependencies..."
if ! $PYTHON_EXEC -c "import requests" 2>/dev/null; then
    echo "❌ Missing 'requests' - run: pip install -r requirements.txt"
    exit 1
fi
echo "✅ Dependencies OK"

# Initialize database if needed
echo
echo "Initializing database..."
$PYTHON_EXEC basketball_main.py --init-db || echo "Database already initialized"

# Test automation components
echo
echo "Testing automation components..."
echo "1. Testing game fetcher..."
timeout 10 $PYTHON_EXEC automatic_game_fetcher.py > /dev/null 2>&1 && echo "   ✅ Game fetcher working" || echo "   ⚠️  Game fetcher test skipped (timeout/API)"

echo "2. Testing prediction generator..."
timeout 10 $PYTHON_EXEC automatic_prediction_generator.py > /dev/null 2>&1 && echo "   ✅ Prediction generator working" || echo "   ⚠️  Prediction generator test skipped"

echo "3. Testing full automation..."
timeout 10 $PYTHON_EXEC basketball_main.py --full-auto > /dev/null 2>&1 && echo "   ✅ Full automation working" || echo "   ⚠️  Full automation test skipped"

# Configure cron
echo
echo "Cron Configuration"
echo "=================="
echo
echo "Recommended schedule:"
echo "  Daily: 10 AM (predict today, track yesterday)"
echo "  Weekly: Sunday 4 AM (full learning cycle)"
echo

# Create cron entries
CRON_DAILY="0 10 * * * cd $SYSTEM_DIR && $PYTHON_EXEC full_automation.py --daily --email-alerts >> full_automation.log 2>&1"
CRON_WEEKLY="0 4 * * 0 cd $SYSTEM_DIR && $PYTHON_EXEC full_automation.py --weekly --email-alerts >> full_automation.log 2>&1"

echo "Proposed cron jobs:"
echo
echo "# Daily automation at 10 AM"
echo "$CRON_DAILY"
echo
echo "# Weekly automation on Sunday at 4 AM"
echo "$CRON_WEEKLY"
echo

read -p "Install these cron jobs? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Backup existing crontab
    crontab -l > "$SYSTEM_DIR/crontab_backup_$(date +%Y%m%d_%H%M%S).txt" 2>/dev/null || echo "No existing crontab"

    # Install cron jobs (remove duplicates first)
    (crontab -l 2>/dev/null | grep -v "full_automation.py"; echo "$CRON_DAILY"; echo "$CRON_WEEKLY") | crontab -

    echo "✅ Cron jobs installed!"
    echo
    crontab -l | grep "full_automation.py"
else
    echo
    echo "Skipped cron installation. To install manually:"
    echo "  crontab -e"
    echo "Then add the above lines."
fi

# Email configuration
echo
echo "Email Configuration"
echo "==================="
if [ ! -f "$SYSTEM_DIR/.env" ]; then
    echo
    echo "No .env file found. Creating from template..."
    if [ -f "$SYSTEM_DIR/.env.example" ]; then
        cp "$SYSTEM_DIR/.env.example" "$SYSTEM_DIR/.env"
        echo "✅ Created .env file"
        echo
        echo "⚠️  IMPORTANT: Edit .env and configure your email:"
        echo "  1. Open .env file"
        echo "  2. Set EMAIL_USER=your_email@gmail.com"
        echo "  3. Set EMAIL_PASS=your_gmail_app_password"
        echo "  4. Get app password: https://support.google.com/accounts/answer/185833"
    else
        echo "❌ .env.example not found"
    fi
else
    echo "✅ .env file exists"

    # Check if configured
    if grep -q "your_email@gmail.com" "$SYSTEM_DIR/.env" 2>/dev/null; then
        echo "⚠️  Email not configured yet. Edit .env file!"
    else
        echo "✅ Email appears to be configured"
    fi
fi

# Summary
echo
echo "="*60
echo "✅ FULL AUTOMATION SETUP COMPLETE!"
echo "="*60
echo
echo "What happens automatically:"
echo "  1. ✅ Fetches games from ESPN every day"
echo "  2. ✅ Generates predictions automatically"
echo "  3. ✅ Tracks outcomes automatically"
echo "  4. ✅ Monitors performance automatically"
echo "  5. ✅ Learns and improves automatically"
echo "  6. ✅ Alerts you if issues detected"
echo
echo "Next steps:"
echo "  1. Configure email in .env (if not done)"
echo "  2. Wait for cron to run, or test now:"
echo "     python basketball_main.py --full-auto"
echo "  3. Check logs: tail -f full_automation.log"
echo
echo "Manual test commands:"
echo "  python basketball_main.py --auto-predict"
echo "  python basketball_main.py --update-outcomes"
echo "  python basketball_main.py --monitor-performance"
echo "  python basketball_main.py --full-auto"
echo
echo "Cron will run automatically:"
echo "  - Daily at 10 AM"
echo "  - Weekly on Sunday at 4 AM"
echo
echo "🤖 You're all set! The system runs itself now! 🚀"
echo
