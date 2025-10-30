#!/usr/bin/env python3
"""
Test script to verify Raspberry Pi setup for palm fruit classification
Run this before deploying the main application
"""
import sys
import os
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def check_python_version():
    """Check Python version"""
    print_header("Python Version Check")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("✅ Python version is sufficient (3.8+)")
        return True
    else:
        print("❌ Python 3.8+ required")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    print_header("Dependency Check")
    
    dependencies = {
        'numpy': 'numpy',
        'PIL': 'Pillow',
        'tkinter': 'python3-tk',
    }
    
    optional_deps = {
        'tflite_runtime': 'tflite-runtime',
        'tensorflow': 'tensorflow',
        'picamera2': 'picamera2',
        'keras': 'keras'
    }
    
    all_good = True
    
    # Check required dependencies
    print("\n📦 Required Dependencies:")
    for module, package in dependencies.items():
        try:
            if module == 'PIL':
                from PIL import Image
            elif module == 'tkinter':
                import tkinter
            else:
                __import__(module)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - Install: pip install {package}")
            all_good = False
    
    # Check optional dependencies
    print("\n📦 Optional Dependencies:")
    tflite_available = False
    keras_available = False
    camera_available = False
    
    for module, package in optional_deps.items():
        try:
            if module == 'tflite_runtime':
                import tflite_runtime.interpreter
                print(f"  ✅ {package} (recommended for TFLite)")
                tflite_available = True
            elif module == 'tensorflow':
                import tensorflow as tf
                print(f"  ✅ {package} (v{tf.__version__})")
                if not tflite_available:
                    print(f"     Can use TFLite via TensorFlow")
                    tflite_available = True
            elif module == 'keras':
                import keras
                print(f"  ✅ {package}")
                keras_available = True
            elif module == 'picamera2':
                from picamera2 import Picamera2
                print(f"  ✅ {package}")
                camera_available = True
        except ImportError:
            print(f"  ⚠️  {package} - Install: pip install {package}")
    
    if not tflite_available:
        print("\n❌ No TFLite support found!")
        print("   Install: pip install tflite-runtime")
        print("   Or: pip install tensorflow")
        all_good = False
    
    if not camera_available:
        print("\n⚠️  Camera support not found (optional)")
        print("   Install: sudo apt-get install python3-picamera2")
    
    return all_good

def check_camera():
    """Check if camera is available"""
    print_header("Camera Check")
    
    try:
        from picamera2 import Picamera2
        
        print("Attempting to initialize camera...")
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (640, 480)}
        )
        picam2.configure(config)
        picam2.start()
        
        # Test capture
        print("Testing image capture...")
        test_file = "/tmp/camera_test.jpg"
        picam2.capture_file(test_file)
        
        picam2.stop()
        picam2.close()
        
        # Check if file was created
        if os.path.exists(test_file):
            file_size = os.path.getsize(test_file)
            print(f"✅ Camera is working! Test image: {file_size} bytes")
            os.remove(test_file)
            return True
        else:
            print("❌ Camera capture failed")
            return False
            
    except Exception as e:
        print(f"❌ Camera error: {str(e)}")
        print("\nTroubleshooting:")
        print("  1. Enable camera: sudo raspi-config > Interface Options > Camera")
        print("  2. Check connection: vcgencmd get_camera")
        print("  3. Reboot: sudo reboot")
        return False

def check_model_file():
    """Check if model file exists"""
    print_header("Model File Check")
    
    model_dir = Path.cwd() / "models"
    
    if not model_dir.exists():
        print(f"❌ Models directory not found: {model_dir}")
        print(f"   Create: mkdir -p {model_dir}")
        return False
    
    print(f"Models directory: {model_dir}")
    
    # Check for model files
    tflite_models = list(model_dir.glob("*.tflite"))
    keras_models = list(model_dir.glob("*.keras")) + list(model_dir.glob("*.h5"))
    
    if tflite_models:
        print("\n✅ TFLite models found:")
        for model in tflite_models:
            size_mb = model.stat().st_size / (1024 * 1024)
            print(f"   - {model.name} ({size_mb:.2f} MB)")
    else:
        print("\n⚠️  No TFLite models found")
    
    if keras_models:
        print("\n✅ Keras models found:")
        for model in keras_models:
            size_mb = model.stat().st_size / (1024 * 1024)
            print(f"   - {model.name} ({size_mb:.2f} MB)")
    else:
        print("\n⚠️  No Keras models found")
    
    if not tflite_models and not keras_models:
        print("\n❌ No model files found!")
        print("\nPlease copy your model file:")
        print("  scp model.tflite pi@raspberrypi.local:~/palm_classifier/models/")
        return False
    
    return True

def test_model_inference():
    """Test model inference"""
    print_header("Model Inference Test")
    
    model_dir = Path.cwd() / "models"
    
    # Find first available model
    tflite_models = list(model_dir.glob("*.tflite"))
    keras_models = list(model_dir.glob("*.keras")) + list(model_dir.glob("*.h5"))
    
    if tflite_models:
        model_path = tflite_models[0]
        is_tflite = True
    elif keras_models:
        model_path = keras_models[0]
        is_tflite = False
    else:
        print("❌ No model found for testing")
        return False
    
    print(f"Testing model: {model_path.name}")
    
    try:
        import numpy as np
        from PIL import Image
        
        if is_tflite:
            # Test TFLite
            try:
                import tflite_runtime.interpreter as tflite
            except ImportError:
                import tensorflow as tf
                tflite = tf.lite
            
            interpreter = tflite.Interpreter(model_path=str(model_path))
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            print(f"  Input shape: {input_details[0]['shape']}")
            print(f"  Output shape: {output_details[0]['shape']}")
            
            # Create dummy input
            input_shape = input_details[0]['shape']
            dummy_input = np.random.rand(*input_shape).astype(np.float32)
            
            # Run inference
            import time
            start = time.time()
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            inference_time = (time.time() - start) * 1000
            
            print(f"\n✅ TFLite inference successful!")
            print(f"  Inference time: {inference_time:.2f}ms")
            print(f"  Output: {output[0]}")
            
        else:
            # Test Keras
            import keras
            
            model = keras.models.load_model(str(model_path))
            print(f"  Input shape: {model.input_shape}")
            print(f"  Output shape: {model.output_shape}")
            
            # Create dummy input
            input_shape = model.input_shape[1:]
            dummy_input = np.random.rand(1, *input_shape).astype(np.float32)
            
            # Run inference
            import time
            start = time.time()
            output = model.predict(dummy_input, verbose=0)
            inference_time = (time.time() - start) * 1000
            
            print(f"\n✅ Keras inference successful!")
            print(f"  Inference time: {inference_time:.2f}ms")
            print(f"  Output: {output[0]}")
        
        # Performance check
        if inference_time < 100:
            print(f"\n✅ Performance is good (<100ms)")
        elif inference_time < 200:
            print(f"\n⚠️  Performance is acceptable (100-200ms)")
        else:
            print(f"\n⚠️  Performance may be slow (>200ms)")
            print("   Consider using TFLite with quantization")
        
        return True
        
    except Exception as e:
        print(f"❌ Inference test failed: {str(e)}")
        return False

def check_system_resources():
    """Check system resources"""
    print_header("System Resources Check")
    
    try:
        # Check CPU info
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            if 'Raspberry Pi 4' in cpuinfo:
                print("✅ Raspberry Pi 4 detected")
            elif 'Raspberry Pi' in cpuinfo:
                print("⚠️  Raspberry Pi detected (not Pi 4)")
            else:
                print("⚠️  Non-Raspberry Pi system")
        
        # Check memory
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemTotal'):
                    mem_kb = int(line.split()[1])
                    mem_gb = mem_kb / (1024 * 1024)
                    print(f"📊 Total RAM: {mem_gb:.2f} GB")
                    
                    if mem_gb >= 4:
                        print("   ✅ RAM is sufficient (4GB+)")
                    elif mem_gb >= 2:
                        print("   ⚠️  RAM is minimal (2GB)")
                    else:
                        print("   ❌ RAM may be insufficient (<2GB)")
                    break
        
        # Check temperature
        try:
            temp_output = os.popen('vcgencmd measure_temp').read()
            temp = float(temp_output.replace("temp=", "").replace("'C\n", ""))
            print(f"🌡️  CPU Temperature: {temp}°C")
            
            if temp < 70:
                print("   ✅ Temperature is good")
            elif temp < 80:
                print("   ⚠️  Temperature is high (consider cooling)")
            else:
                print("   ❌ Temperature is critical (add cooling!)")
        except:
            print("⚠️  Could not read temperature")
        
        # Check disk space
        stat = os.statvfs('/')
        available_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        print(f"💾 Available disk space: {available_gb:.2f} GB")
        
        if available_gb > 2:
            print("   ✅ Disk space is sufficient")
        else:
            print("   ⚠️  Low disk space")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Could not check system resources: {str(e)}")
        return True

def check_directory_structure():
    """Check directory structure"""
    print_header("Directory Structure Check")
    
    base_dir = Path.cwd()
    
    required_dirs = [
        "models",
        "results",
        "results/captured_images"
    ]
    
    all_good = True
    
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            print(f"  ✅ {dir_name}/")
        else:
            print(f"  ❌ {dir_name}/ - Creating...")
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"     Created: {dir_path}")
    
    return all_good

def main():
    """Run all tests"""
    print("\n")
    print("🍓" * 30)
    print("  RASPBERRY PI SETUP TEST")
    print("  Palm Fruit Classification System")
    print("🍓" * 30)
    
    results = []
    
    # Run all tests
    results.append(("Python Version", check_python_version()))
    results.append(("Dependencies", check_dependencies()))
    results.append(("Directory Structure", check_directory_structure()))
    results.append(("System Resources", check_system_resources()))
    results.append(("Camera", check_camera()))
    results.append(("Model Files", check_model_file()))
    results.append(("Model Inference", test_model_inference()))
    
    # Summary
    print_header("TEST SUMMARY")
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name:25s} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    
    if all_passed:
        print("\n🎉 All tests passed! System is ready.")
        print("\nYou can now run the application:")
        print("  python3 raspberry_pi_camera_app.py")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\nRefer to RASPBERRY_PI_INSTALLATION.md for help.")
    
    print("\n")

if __name__ == "__main__":
    main()