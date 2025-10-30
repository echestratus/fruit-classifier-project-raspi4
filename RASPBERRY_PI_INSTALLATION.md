# 🍓 Instalasi & Setup Raspberry Pi 4

## 📋 Prerequisites

- Raspberry Pi 4 (2GB RAM minimum, 4GB+ recommended)
- Raspberry Pi Camera Module atau USB Camera
- Raspberry Pi OS (Bullseye atau lebih baru)
- Internet connection untuk instalasi

---

## 🔧 Step 1: Update System

```bash
# Update package list
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
sudo apt-get install -y python3-pip python3-venv
sudo apt-get install -y libatlas-base-dev libhdf5-dev
sudo apt-get install -y python3-tk  # Untuk GUI Tkinter
```

---

## 📷 Step 2: Enable Camera

### Untuk Raspberry Pi Camera Module:

```bash
# Enable camera interface
sudo raspi-config
# Pilih: Interface Options > Camera > Enable

# Reboot
sudo reboot
```

### Test Camera:

```bash
# Install picamera2
sudo apt-get install -y python3-picamera2

# Test capture
python3 -c "from picamera2 import Picamera2; cam = Picamera2(); cam.start(); cam.capture_file('test.jpg'); print('✅ Camera OK')"
```

---

## 🐍 Step 3: Setup Python Environment

```bash
# Create project directory
mkdir -p ~/palm_classifier
cd ~/palm_classifier

# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

---

## 📦 Step 4: Install Dependencies

### Option A: Install TensorFlow Lite Runtime (RECOMMENDED - Lightweight)

```bash
# Install TFLite Runtime
pip install tflite-runtime

# Install other dependencies
pip install pillow numpy
pip install picamera2  # If not already installed
```

### Option B: Install Full TensorFlow (Heavier, but supports both .keras and .tflite)

```bash
# Install TensorFlow for Raspberry Pi
pip install tensorflow

# Install other dependencies
pip install pillow numpy
pip install picamera2
```

### Verify Installation:

```bash
# Test TFLite Runtime
python3 -c "import tflite_runtime.interpreter as tflite; print('✅ TFLite Runtime OK')"

# Or test TensorFlow
python3 -c "import tensorflow as tf; print('✅ TensorFlow version:', tf.__version__)"

# Test GUI
python3 -c "import tkinter; print('✅ Tkinter OK')"

# Test PIL
python3 -c "from PIL import Image; print('✅ PIL OK')"
```

---

## 📁 Step 5: Setup Project Structure

```bash
cd ~/palm_classifier

# Create directory structure
mkdir -p models
mkdir -p results/captured_images

# Directory structure:
# palm_classifier/
# ├── raspberry_pi_camera_app.py
# ├── models/
# │   └── palm_classifier_float16.tflite
# └── results/
#     ├── captured_images/
#     └── classification_results.csv (auto-generated)
```

---

## 📥 Step 6: Transfer Files from Development Machine

### Transfer Model File:

```bash
# From your development machine (Windows/Linux/Mac):

# Option 1: Using SCP
scp models/palm_classifier_float16.tflite pi@raspberrypi.local:~/palm_classifier/models/

# Option 2: Using USB drive
# 1. Copy model file to USB drive
# 2. Insert USB to Raspberry Pi
# 3. Mount and copy:
sudo mount /dev/sda1 /mnt
cp /mnt/palm_classifier_float16.tflite ~/palm_classifier/models/
sudo umount /mnt

# Option 3: Download from cloud storage (if uploaded)
# wget https://your-storage-url/palm_classifier_float16.tflite -P models/
```

### Transfer Application Script:

```bash
# From development machine:
scp raspberry_pi_camera_app.py pi@raspberrypi.local:~/palm_classifier/

# Or manually copy the script content
```

---

## 🚀 Step 7: Run the Application

```bash
cd ~/palm_classifier

# Activate virtual environment if created
source venv/bin/activate

# Run application
python3 raspberry_pi_camera_app.py
```

### Expected Output:

```
============================================================
🌴 PALM FRUIT CLASSIFICATION - RASPBERRY PI
============================================================
Model: /home/pi/palm_classifier/models/palm_classifier_float16.tflite
Classes: ['sawit_mentah', 'sawit_matang']
Confidence threshold: 0.7
============================================================

📥 Loading model from /home/pi/palm_classifier/models/palm_classifier_float16.tflite
   Input shape: [  1 224 224   3]
   Input type: <class 'numpy.float32'>
✅ Model loaded successfully!

✅ Camera initialized successfully
```

---

## 🎮 Using the Application

### GUI Controls:

1. **📸 Capture from Camera** - Ambil foto dari kamera
2. **📂 Select from File** - Pilih gambar dari file system
3. **🗑️ Clear** - Hapus hasil prediksi

### Results Display:

- **File:** Path ke gambar yang dianalisis
- **Result:** Klasifikasi (Sawit Matang/Mentah/Unknown)
- **Confidence:** Tingkat kepercayaan prediksi (0-100%)
- **Probabilities:** Probabilitas untuk setiap kelas
- **Entropy:** Ukuran ketidakpastian model
- **Inference Time:** Waktu prediksi dalam milliseconds

### Results Logging:

Semua hasil prediksi otomatis disimpan di: `results/classification_results.csv`

Format CSV:
```csv
timestamp,image_path,predicted_class,confidence,is_unknown,entropy,inference_time_ms,sawit_mentah_prob,sawit_matang_prob
2025-10-29 10:30:15,/home/pi/palm_classifier/results/captured_images/capture_20251029_103015.jpg,sawit_matang,0.9234,False,0.2145,67.34,0.0766,0.9234
```

---

## 🔧 Configuration & Customization

### Adjust Confidence Threshold:

Edit `raspberry_pi_camera_app.py`:

```python
class Config:
    # Change threshold (0.0 to 1.0)
    CONFIDENCE_THRESHOLD = 0.70  # Default
    # Lower = More sensitive (more unknowns)
    # Higher = More strict (fewer unknowns)
```

### Change Model:

```python
class Config:
    # Use TFLite model (recommended)
    MODEL_PATH = MODELS_DIR / "palm_classifier_float16.tflite"
    
    # Or use Keras model (requires full TensorFlow)
    # MODEL_PATH = MODELS_DIR / "palm_classifier_final.keras"
```

### Adjust Camera Settings:

```python
class Config:
    CAMERA_RESOLUTION = (640, 480)  # Default
    # Higher resolution = Better quality, slower
    # Try: (1280, 720) or (1920, 1080)
    
    FRAME_RATE = 30  # FPS for preview
```

---

## 🐛 Troubleshooting

### Issue 1: Camera Not Detected

```bash
# Check if camera is enabled
vcgencmd get_camera

# Should output: supported=1 detected=1

# If not, enable via raspi-config
sudo raspi-config
# Interface Options > Camera > Enable
sudo reboot
```

### Issue 2: "No module named 'picamera2'"

```bash
# Install picamera2
sudo apt-get install -y python3-picamera2

# Or via pip
pip install picamera2
```

### Issue 3: "No module named 'tflite_runtime'"

```bash
# Option 1: Install TFLite Runtime
pip install tflite-runtime

# Option 2: Install full TensorFlow
pip install tensorflow
```

### Issue 4: Slow Inference (>500ms)

**Solutions:**
1. Use TFLite model instead of Keras
2. Use float16 or dynamic quantization
3. Reduce camera resolution
4. Close other applications

```bash
# Check CPU usage
htop

# Check temperature (should be < 80°C)
vcgencmd measure_temp

# If overheating, add cooling
```

### Issue 5: "Model not found"

```bash
# Check if model file exists
ls -lh ~/palm_classifier/models/

# Verify file permissions
chmod 644 ~/palm_classifier/models/*.tflite

# Check file size (should be ~3-10 MB)
du -h ~/palm_classifier/models/*.tflite
```

### Issue 6: GUI Window Not Showing

```bash
# Enable X11 forwarding if using SSH
ssh -X pi@raspberrypi.local

# Or run directly on Raspberry Pi desktop
# Connect monitor, keyboard, mouse to RPi
# Open terminal and run the application
```

### Issue 7: Low FPS / Laggy Preview

```python
# Reduce preview resolution in Config
PREVIEW_SIZE = (160, 120)  # Instead of (320, 240)

# Reduce camera resolution
CAMERA_RESOLUTION = (640, 480)  # Instead of (1280, 720)

# Increase frame delay
self.root.after(50, self.update_camera_frame)  # 20 FPS instead of 30
```

---

## ⚡ Performance Optimization

### 1. Overclock Raspberry Pi 4 (Optional)

**⚠️ Warning:** May void warranty and reduce lifespan. Ensure adequate cooling!

```bash
# Edit config
sudo nano /boot/config.txt

# Add these lines:
over_voltage=6
arm_freq=2000

# Save and reboot
sudo reboot

# Check current frequency
vcgencmd get_config arm_freq
```

### 2. Use Lightweight Desktop Environment

```bash
# Switch to LXDE (lighter than default)
sudo apt-get install lxde
# Select LXDE at login
```

### 3. Disable Unnecessary Services

```bash
# Disable Bluetooth (if not needed)
sudo systemctl disable bluetooth

# Disable WiFi (if using Ethernet)
sudo systemctl disable wpa_supplicant
```

### 4. Monitor Performance

```bash
# Real-time monitoring script
watch -n 1 'vcgencmd measure_temp && vcgencmd measure_clock arm'
```

---

## 📊 Benchmark Results (Expected)

### Raspberry Pi 4 Model B (4GB RAM)

| Model | Size | Inference Time | FPS | CPU Usage |
|-------|------|----------------|-----|-----------|
| Keras (.keras) | ~14 MB | 200-300ms | 3-5 | 80-100% |
| TFLite Float16 | ~7 MB | 50-80ms | 12-20 | 60-80% |
| TFLite Dynamic | ~4 MB | 40-60ms | 16-25 | 50-70% |
| TFLite Int8 | ~4 MB | 30-50ms | 20-30 | 40-60% |

**Recommended:** TFLite Float16 (best balance)

---

## 🚀 Auto-Start on Boot (Optional)

### Create Systemd Service:

```bash
# Create service file
sudo nano /etc/systemd/system/palm-classifier.service
```

Add this content:

```ini
[Unit]
Description=Palm Fruit Classifier Application
After=multi-user.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/palm_classifier
Environment="DISPLAY=:0"
Environment="XAUTHORITY=/home/pi/.Xauthority"
ExecStart=/usr/bin/python3 /home/pi/palm_classifier/raspberry_pi_camera_app.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=graphical.target
```

Enable and start:

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service
sudo systemctl enable palm-classifier.service

# Start service
sudo systemctl start palm-classifier.service

# Check status
sudo systemctl status palm-classifier.service

# View logs
journalctl -u palm-classifier.service -f
```

---

## 🔐 Security Recommendations

### 1. Change Default Password

```bash
passwd
# Enter new password
```

### 2. Enable Firewall (if connected to network)

```bash
sudo apt-get install ufw
sudo ufw enable
sudo ufw allow ssh
```

### 3. Keep System Updated

```bash
# Create update script
nano ~/update.sh
```

Add:

```bash
#!/bin/bash
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get autoremove -y
sudo apt-get autoclean
```

```bash
# Make executable
chmod +x ~/update.sh

# Run weekly
crontab -e
# Add: 0 2 * * 0 /home/pi/update.sh
```

---

## 📱 Remote Access

### 1. Enable VNC (for GUI access)

```bash
sudo raspi-config
# Interface Options > VNC > Enable

# Install VNC Viewer on your computer
# Connect to: raspberrypi.local
```

### 2. Access via SSH

```bash
# From your computer
ssh pi@raspberrypi.local

# Run application with X11 forwarding
ssh -X pi@raspberrypi.local
cd ~/palm_classifier
python3 raspberry_pi_camera_app.py
```

---

## 📈 Monitoring & Logging

### View Real-time Logs:

```bash
# Watch results CSV
tail -f ~/palm_classifier/results/classification_results.csv

# Monitor with timestamps
watch -n 1 'tail -5 ~/palm_classifier/results/classification_results.csv'
```

### Analyze Results:

```bash
# Count predictions per class
awk -F',' 'NR>1 {print $3}' results/classification_results.csv | sort | uniq -c

# Calculate average confidence
awk -F',' 'NR>1 {sum+=$4; count++} END {print "Avg Confidence:", sum/count}' results/classification_results.csv

# Count unknown predictions
awk -F',' 'NR>1 && $5=="True" {count++} END {print "Unknown count:", count}' results/classification_results.csv
```

---

## 🎯 Testing Checklist

Before deployment, test:

- [ ] Camera capture works
- [ ] File selection works
- [ ] Model loads successfully
- [ ] Predictions are accurate (>85%)
- [ ] Unknown detection works (test with non-palm images)
- [ ] Results are logged to CSV
- [ ] GUI is responsive
- [ ] Inference time < 100ms
- [ ] No memory leaks (run for 1+ hours)
- [ ] Works after reboot

---

## 📞 Support & Resources

### Official Documentation:
- Raspberry Pi: https://www.raspberrypi.com/documentation/
- TensorFlow Lite: https://www.tensorflow.org/lite
- Picamera2: https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf

### Community:
- Raspberry Pi Forums: https://forums.raspberrypi.com/
- TensorFlow Forum: https://discuss.tensorflow.org/

---

## 🎉 Success!

Your Palm Fruit Classifier is now running on Raspberry Pi 4!

Next steps:
1. Test with various palm fruit samples
2. Monitor accuracy and adjust threshold if needed
3. Collect more data to improve model
4. Consider adding features:
   - Email/SMS notifications
   - Database storage
   - Web interface
   - Multiple camera support
   - Batch processing

**Happy Classifying! 🌴**