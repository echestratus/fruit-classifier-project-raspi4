#!/usr/bin/env python3
"""
Simplified Palm Fruit Classification App for Raspberry Pi
Use this if the main app has GUI issues
"""
import os
import time
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np

# Camera import
try:
    from picamera2 import Picamera2
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False

# TFLite import
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tf
    tflite = tf.lite


class SimpleConfig:
    """Simple configuration"""
    BASE_DIR = Path(__file__).parent
    MODEL_PATH = BASE_DIR / "models" / "palm_classifier_float16.tflite"
    RESULTS_DIR = BASE_DIR / "results"
    CAPTURED_DIR = RESULTS_DIR / "captured_images"
    
    IMG_SIZE = (224, 224)
    CLASS_NAMES = ['sawit_mentah', 'sawit_matang']
    CONFIDENCE_THRESHOLD = 0.70
    
    @classmethod
    def setup(cls):
        cls.RESULTS_DIR.mkdir(exist_ok=True)
        cls.CAPTURED_DIR.mkdir(exist_ok=True)


class SimplePalmClassifier:
    def __init__(self, root):
        self.root = root
        self.root.title("Palm Fruit Classifier")
        
        # Setup
        SimpleConfig.setup()
        
        # Variables
        self.current_image = None
        self.camera = None
        self.interpreter = None
        
        # Load model
        self.load_model()
        
        # Init camera
        if CAMERA_AVAILABLE:
            self.init_camera()
        
        # Build UI
        self.build_ui()
        
        # Start camera loop
        if self.camera:
            self.update_camera()
    
    def load_model(self):
        """Load TFLite model"""
        try:
            print(f"Loading model: {SimpleConfig.MODEL_PATH}")
            self.interpreter = tflite.Interpreter(model_path=str(SimpleConfig.MODEL_PATH))
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print("✅ Model loaded")
        except Exception as e:
            messagebox.showerror("Error", f"Model load failed:\n{e}")
    
    def init_camera(self):
        """Initialize camera"""
        try:
            self.camera = Picamera2()
            config = self.camera.create_preview_configuration(
                main={"format": "RGB888", "size": (640, 480)}
            )
            self.camera.configure(config)
            self.camera.start()
            print("✅ Camera ready")
        except Exception as e:
            print(f"Camera failed: {e}")
            self.camera = None
    
    def build_ui(self):
        """Build simple UI with scrollbar"""
        # Make window adjustable
        self.root.geometry("750x600")
        self.root.minsize(600, 400)
        
        # Create canvas with scrollbar
        canvas = tk.Canvas(self.root, highlightthickness=0)
        scrollbar = tk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        
        # Scrollable frame
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        # Update canvas window width when resized
        def on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
        
        canvas.bind("<Configure>", on_canvas_configure)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Enable mousewheel scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def on_mousewheel_linux(event):
            if event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")
        
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        canvas.bind_all("<Button-4>", on_mousewheel_linux)
        canvas.bind_all("<Button-5>", on_mousewheel_linux)
        
        # Build UI components in scrollable_frame
        root_frame = scrollable_frame
        
        # Title
        title = tk.Label(root_frame, text="🌴 Palm Fruit Classifier", 
                        font=("Arial", 16, "bold"), pady=10)
        title.pack()
        
        # Camera preview
        cam_frame = tk.LabelFrame(root_frame, text="Camera Preview", font=("Arial", 12, "bold"))
        cam_frame.pack(padx=10, pady=5, fill=tk.BOTH)
        
        self.cam_label = tk.Label(cam_frame, bg="black", width=320, height=240)
        self.cam_label.pack(padx=10, pady=10)
        
        # Captured image
        img_frame = tk.LabelFrame(root_frame, text="Captured Image", font=("Arial", 12, "bold"))
        img_frame.pack(padx=10, pady=5, fill=tk.BOTH)
        
        self.img_label = tk.Label(img_frame, bg="gray", width=320, height=240)
        self.img_label.pack(padx=10, pady=10)
        
        # Buttons Frame
        btn_frame = tk.Frame(root_frame)
        btn_frame.pack(pady=10)
        
        # Buttons
        if self.camera:
            tk.Button(btn_frame, text="📸 Capture", command=self.capture,
                     bg="green", fg="white", font=("Arial", 12), 
                     width=15, height=2).grid(row=0, column=0, padx=5)
        
        tk.Button(btn_frame, text="📂 Open File", command=self.open_file,
                 bg="blue", fg="white", font=("Arial", 12),
                 width=15, height=2).grid(row=0, column=1, padx=5)
        
        tk.Button(btn_frame, text="🗑️ Clear", command=self.clear,
                 bg="red", fg="white", font=("Arial", 12),
                 width=15, height=2).grid(row=0, column=2, padx=5)
        
        # Results Frame
        result_frame = tk.LabelFrame(root_frame, text="Results", font=("Arial", 12, "bold"))
        result_frame.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        
        # Result labels
        self.result_text = tk.Label(result_frame, text="No prediction yet",
                                    font=("Arial", 18, "bold"), fg="gray", pady=10)
        self.result_text.pack()
        
        self.conf_text = tk.Label(result_frame, text="Confidence: -",
                                  font=("Arial", 14), pady=5)
        self.conf_text.pack()
        
        # Probabilities
        self.prob_text = tk.Text(result_frame, height=5, width=50, 
                                font=("Courier", 10))
        self.prob_text.pack(pady=10, padx=10)
        
        # Status
        self.status = tk.Label(root_frame, text="Ready", relief=tk.SUNKEN,
                              anchor=tk.W, font=("Arial", 10))
        self.status.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
    
    def update_camera(self):
        """Update camera preview"""
        if not self.camera:
            return
        
        try:
            frame = self.camera.capture_array()
            img = Image.fromarray(frame)
            img = img.resize((320, 240))
            photo = ImageTk.PhotoImage(img)
            self.cam_label.config(image=photo)
            self.cam_label.image = photo
        except:
            pass
        
        self.root.after(33, self.update_camera)
    
    def capture(self):
        """Capture from camera"""
        if not self.camera:
            messagebox.showwarning("Warning", "Camera not available")
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = SimpleConfig.CAPTURED_DIR / f"capture_{timestamp}.jpg"
            self.camera.capture_file(str(filename))
            self.current_image = str(filename)
            self.display_image()
            self.predict()
        except Exception as e:
            messagebox.showerror("Error", f"Capture failed:\n{e}")
    
    def open_file(self):
        """Open image file"""
        filepath = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
        )
        if filepath:
            self.current_image = filepath
            self.display_image()
            self.predict()
    
    def display_image(self):
        """Display current image"""
        if not self.current_image:
            return
        
        img = Image.open(self.current_image)
        img = img.resize((320, 240))
        photo = ImageTk.PhotoImage(img)
        self.img_label.config(image=photo)
        self.img_label.image = photo
    
    def predict(self):
        """Run prediction"""
        if not self.current_image or not self.interpreter:
            return
        
        try:
            self.status.config(text="Predicting...", fg="orange")
            self.root.update()
            
            # Preprocess
            img = Image.open(self.current_image).convert('RGB')
            img = img.resize(SimpleConfig.IMG_SIZE)
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            start = time.time()
            self.interpreter.set_tensor(self.input_details[0]['index'], img_array)
            self.interpreter.invoke()
            predictions = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            inference_time = (time.time() - start) * 1000
            
            # Get result
            pred_idx = np.argmax(predictions)
            pred_class = SimpleConfig.CLASS_NAMES[pred_idx]
            confidence = float(predictions[pred_idx])
            
            # Check unknown
            if confidence < SimpleConfig.CONFIDENCE_THRESHOLD:
                result = "❓ UNKNOWN"
                color = "orange"
                display_class = f"(Top: {pred_class})"
            else:
                result = f"✅ {pred_class.replace('_', ' ').title()}"
                color = "green" if pred_class == 'sawit_matang' else "red"
                display_class = ""
            
            # Update UI
            self.result_text.config(text=f"{result} {display_class}", fg=color)
            self.conf_text.config(text=f"Confidence: {confidence:.2%}")
            
            # Update probabilities
            self.prob_text.delete(1.0, tk.END)
            prob_str = ""
            for i, name in enumerate(SimpleConfig.CLASS_NAMES):
                prob = predictions[i]
                bar = "█" * int(prob * 30)
                prob_str += f"{name:15s}: {prob:.4f} {bar}\n"
            prob_str += f"\nInference: {inference_time:.1f}ms"
            self.prob_text.insert(1.0, prob_str)
            
            # Log
            self.log_result(pred_class, confidence, inference_time)
            
            self.status.config(text=f"Done ({inference_time:.1f}ms)", fg="green")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed:\n{e}")
            self.status.config(text="Prediction failed", fg="red")
    
    def log_result(self, pred_class, confidence, inference_time):
        """Log to CSV"""
        log_file = SimpleConfig.RESULTS_DIR / "results.csv"
        
        if not log_file.exists():
            with open(log_file, 'w') as f:
                f.write("timestamp,image,class,confidence,inference_ms\n")
        
        with open(log_file, 'a') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp},{self.current_image},{pred_class},{confidence:.4f},{inference_time:.2f}\n")
    
    def clear(self):
        """Clear results"""
        self.current_image = None
        self.img_label.config(image='')
        self.result_text.config(text="No prediction yet", fg="gray")
        self.conf_text.config(text="Confidence: -")
        self.prob_text.delete(1.0, tk.END)
        self.status.config(text="Cleared", fg="blue")
    
    def cleanup(self):
        """Cleanup on exit"""
        if self.camera:
            self.camera.stop()
            self.camera.close()
        self.root.destroy()


def main():
    """Main entry"""
    print("="*60)
    print("🌴 SIMPLE PALM FRUIT CLASSIFIER")
    print("="*60)
    print(f"Model: {SimpleConfig.MODEL_PATH}")
    print(f"Camera: {'Available' if CAMERA_AVAILABLE else 'Not available'}")
    print("="*60)
    
    if not SimpleConfig.MODEL_PATH.exists():
        print(f"\n❌ Model not found: {SimpleConfig.MODEL_PATH}")
        print("Please copy your model file to models/ directory")
        return
    
    root = tk.Tk()
    app = SimplePalmClassifier(root)
    
    # Store app reference to prevent garbage collection
    root.app_instance = app
    
    # Set cleanup on window close
    root.protocol("WM_DELETE_WINDOW", app.cleanup)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped")
        app.cleanup()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if hasattr(app, 'cleanup'):
            app.cleanup()


if __name__ == "__main__":
    main()