import sys
import os
import platform
import subprocess

print("=== Python Environment Check ===")
print(f"Python version: {platform.python_version()}")
print(f"Python executable: {sys.executable}")
print(f"Python path: {sys.path}")

print("\n=== Required Packages ===")
required_packages = [
    "flask", "numpy", "matplotlib", "opencv-python", 
    "torch", "open3d", "scikit-image"
]

for package in required_packages:
    try:
        __import__(package)
        version = __import__(package).__version__
        print(f"✅ {package}: {version}")
    except ImportError:
        print(f"❌ {package}: Not installed")
    except AttributeError:
        print(f"✅ {package}: Installed (version not available)")

print("\n=== Flask Server Test ===")
try:
    import flask
    print("Flask version:", flask.__version__)
    from flask import Flask
    app = Flask(__name__)
    print("✅ Flask app created successfully")
except Exception as e:
    print(f"❌ Flask error: {e}")

print("\n=== Networking Check ===")
try:
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', 5000))
    if result == 0:
        print("⚠️ Port 5000 is already in use. Stop any running servers first.")
    else:
        print("✅ Port 5000 is available")
    sock.close()
except Exception as e:
    print(f"❌ Network error: {e}")

print("\nRun this command to start the server:")
print("cd c:\\Users\\USER\\Desktop\\vit_project\\model_folder && python app.py")
