#!/usr/bin/env python3
"""
Main entry point for the Trading Signal Bot
Run this from the peak_detection directory: python run_bot.py
"""
import os
import sys
import subprocess

def main():
    """Main entry point that delegates to the bot service"""
    # Get the current directory (should be peak_detection)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    bot_service_dir = os.path.join(current_dir, 'bot_service')
    
    # Check if bot_service directory exists
    if not os.path.exists(bot_service_dir):
        print("[ERROR] bot_service directory not found!")
        print(f"   Expected: {bot_service_dir}")
        return 1
    
    # Check if we're running in Docker (no virtual environment needed)
    is_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_ENV') == '1'
    
    if is_docker:
        # In Docker, use system Python
        python_exe = sys.executable
        print("[INFO] Running in Docker container - using system Python")
    else:
        # Check if virtual environment exists
        venv_dir = os.path.join(bot_service_dir, '.venv')
        if not os.path.exists(venv_dir):
            print("[ERROR] Virtual environment not found!")
            print(f"   Expected: {venv_dir}")
            print("\n[INFO] To create virtual environment:")
            print("   cd bot_service")
            print("   python -m venv .venv")
            print("   .venv\\Scripts\\Activate.ps1")
            print("   pip install -r requirements.txt")
            return 1
        
        # Path to Python executable in virtual environment
        if os.name == 'nt':  # Windows
            python_exe = os.path.join(venv_dir, 'Scripts', 'python.exe')
        else:  # Unix/Linux/Mac
            python_exe = os.path.join(venv_dir, 'bin', 'python')
        
        if not os.path.exists(python_exe):
            print(f"[ERROR] Python executable not found in virtual environment!")
            print(f"   Expected: {python_exe}")
            return 1
    
    # Check if local_settings.py exists
    local_settings_path = os.path.join(current_dir, 'local_settings.py')
    if not os.path.exists(local_settings_path):
        print("[WARNING] local_settings.py not found!")
        print(f"   Expected: {local_settings_path}")
        print("\n[INFO] Please create local_settings.py with your credentials")
    
    # Path to the actual bot script
    bot_script = os.path.join(bot_service_dir, 'run_bot.py')
    
    print("[STARTING] Trading Signal Bot...")
    print(f"   Working directory: {current_dir}")
    print(f"   Bot service: {bot_service_dir}")
    print(f"   Python: {python_exe}")
    print("-" * 50)
    
    # Change to bot_service directory and run the bot
    try:
        # Run the bot with the virtual environment Python
        result = subprocess.run([python_exe, bot_script], 
                              cwd=bot_service_dir,
                              check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\n[STOPPED] Bot stopped by user")
        return 0
    except Exception as e:
        print(f"[ERROR] Error running bot: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
