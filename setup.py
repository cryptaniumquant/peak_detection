#!/usr/bin/env python3
"""
Setup script for Trading Signal Bot
Run this once to set up the environment: python setup.py
"""
import os
import sys
import subprocess
import shutil

def run_command(cmd, cwd=None, shell=True):
    """Run a command and return success status"""
    try:
        print(f"🔧 Running: {cmd}")
        result = subprocess.run(cmd, cwd=cwd, shell=shell, check=True, 
                              capture_output=False, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Trading Signal Bot Setup")
    print("=" * 50)
    
    # Get current directory (should be peak_detection)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    bot_service_dir = os.path.join(current_dir, 'bot_service')
    
    print(f"📁 Project directory: {current_dir}")
    print(f"📁 Bot service directory: {bot_service_dir}")
    
    # Check if bot_service exists
    if not os.path.exists(bot_service_dir):
        print("❌ Error: bot_service directory not found!")
        return 1
    
    # Step 1: Create virtual environment
    print("\n📦 Step 1: Creating virtual environment...")
    venv_dir = os.path.join(bot_service_dir, '.venv')
    
    if os.path.exists(venv_dir):
        print("✅ Virtual environment already exists")
    else:
        if not run_command([sys.executable, '-m', 'venv', '.venv'], cwd=bot_service_dir, shell=False):
            print("❌ Failed to create virtual environment")
            return 1
        print("✅ Virtual environment created")
    
    # Step 2: Install dependencies
    print("\n📦 Step 2: Installing dependencies...")
    if os.name == 'nt':  # Windows
        pip_exe = os.path.join(venv_dir, 'Scripts', 'pip.exe')
        python_exe = os.path.join(venv_dir, 'Scripts', 'python.exe')
    else:  # Unix/Linux/Mac
        pip_exe = os.path.join(venv_dir, 'bin', 'pip')
        python_exe = os.path.join(venv_dir, 'bin', 'python')
    
    requirements_file = os.path.join(bot_service_dir, 'requirements.txt')
    if not os.path.exists(requirements_file):
        print("❌ requirements.txt not found!")
        return 1
    
    if not run_command([pip_exe, 'install', '-r', 'requirements.txt'], 
                      cwd=bot_service_dir, shell=False):
        print("❌ Failed to install dependencies")
        return 1
    print("✅ Dependencies installed")
    
    # Step 3: Install asyncmy for async database support
    print("\n📦 Step 3: Installing async database driver...")
    if not run_command([pip_exe, 'install', 'asyncmy==0.2.9'], 
                      cwd=bot_service_dir, shell=False):
        print("❌ Failed to install asyncmy")
        return 1
    print("✅ Async database driver installed")
    
    # Step 4: Check configuration files
    print("\n⚙️  Step 4: Checking configuration...")
    
    # Check local_settings.py
    local_settings_path = os.path.join(current_dir, 'local_settings.py')
    if os.path.exists(local_settings_path):
        print("✅ local_settings.py found")
    else:
        print("⚠️  local_settings.py not found - you'll need to create it")
        print("   This file should contain your database and Telegram credentials")
    
    # Check strategy_thresholds.json
    thresholds_path = os.path.join(current_dir, 'strategy_thresholds.json')
    if os.path.exists(thresholds_path):
        print("✅ strategy_thresholds.json found")
    else:
        print("⚠️  strategy_thresholds.json not found")
    
    # Step 5: Test imports
    print("\n🧪 Step 5: Testing imports...")
    test_cmd = [python_exe, '-c', 
                'import sys; sys.path.insert(0, ".."); '
                'from config import load_settings; '
                'print("✅ Config import successful")']
    
    if run_command(test_cmd, cwd=bot_service_dir, shell=False):
        print("✅ All imports working")
    else:
        print("❌ Import test failed")
        return 1
    
    # Success!
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Create/update local_settings.py with your credentials")
    print("2. Update strategy_thresholds.json with your strategy thresholds")
    print("3. Run the bot: python run_bot.py")
    print("\n💡 The bot will run from the main peak_detection directory")
    print("   No need to activate virtual environment manually!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
