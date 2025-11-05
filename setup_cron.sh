#!/bin/bash
# Setup cron jobs for self-learning basketball system

set -e

echo "🧠 Basketball Self-Learning System - Cron Setup"
echo "==============================================="
echo

# Get the current directory
SYSTEM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "System directory: $SYSTEM_DIR"

# Find Python executable (prefer venv if available)
if [ -f "$SYSTEM_DIR/venv/bin/python" ]; then
    PYTHON_EXEC="$SYSTEM_DIR/venv/bin/python"
    echo "Using Python from venv: $PYTHON_EXEC"
elif command -v python3 &> /dev/null; then
    PYTHON_EXEC=$(command -v python3)
    echo "Using system Python: $PYTHON_EXEC"
else
    echo "❌ Error: Python not found"
    exit 1
fi

# Verify Python can run the scripts
echo
echo "Testing Python scripts..."
if ! $PYTHON_EXEC -c "import sys; print(f'Python {sys.version}')"; then
    echo "❌ Python test failed"
    exit 1
fi
echo "✅ Python verified"

# Create cron entries
CRON_DAILY="0 2 * * * cd $SYSTEM_DIR && $PYTHON_EXEC scheduled_self_improvement.py --daily --email-alerts >> self_improvement.log 2>&1"
CRON_WEEKLY="0 3 * * 0 cd $SYSTEM_DIR && $PYTHON_EXEC scheduled_self_improvement.py --weekly --email-alerts >> self_improvement.log 2>&1"

echo
echo "Proposed cron entries:"
echo "======================"
echo
echo "# Daily self-learning (2 AM every day)"
echo "$CRON_DAILY"
echo
echo "# Weekly learning cycle (3 AM every Sunday)"
echo "$CRON_WEEKLY"
echo

# Ask for confirmation
read -p "Install these cron jobs? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted. You can manually add cron jobs with:"
    echo "  crontab -e"
    exit 0
fi

# Backup existing crontab
echo
echo "Backing up current crontab..."
crontab -l > "$SYSTEM_DIR/crontab_backup_$(date +%Y%m%d_%H%M%S).txt" 2>/dev/null || echo "No existing crontab"

# Add new cron jobs (avoiding duplicates)
echo
echo "Installing cron jobs..."
(crontab -l 2>/dev/null | grep -v "scheduled_self_improvement.py"; echo "$CRON_DAILY"; echo "$CRON_WEEKLY") | crontab -

echo "✅ Cron jobs installed successfully!"
echo

# Display installed cron jobs
echo "Current cron jobs:"
echo "=================="
crontab -l | grep -A 2 "scheduled_self_improvement.py" || echo "No basketball cron jobs found"
echo

# Test run
echo "Would you like to test the self-learning system now? (y/n) "
read -p "> " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo
    echo "Running test..."
    echo "==============="
    cd "$SYSTEM_DIR"
    $PYTHON_EXEC basketball_main.py --run-learning-cycle
fi

echo
echo "✅ Setup complete!"
echo
echo "Next steps:"
echo "1. Check logs: tail -f $SYSTEM_DIR/self_improvement.log"
echo "2. Configure email in .env (optional)"
echo "3. Wait for daily/weekly runs or test manually"
echo
echo "Manual test commands:"
echo "  python basketball_main.py --update-outcomes"
echo "  python basketball_main.py --monitor-performance"
echo "  python basketball_main.py --run-learning-cycle"
echo
