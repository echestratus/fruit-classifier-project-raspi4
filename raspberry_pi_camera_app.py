"""
Palm Fruit Classification Camera Application for Raspberry Pi 4
Supports both Keras (.keras) and TFLite (.tflite) models
Includes unknown detection mechanism
"""
import os
import time
import json
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np

# Try to import picamera2 for RPi Camera
try:
    from picamera2 import Picamera2
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False
    print("⚠️  Picamera2 not available. Camera capture will be disabled.")

# Try to import TFLite runtime (lightweight) or fallback to TensorFlow
try:
    import tflite_runtime.interpreter as tflite
    USE_TFLITE_RUNTIME = True
except ImportError:
    try:
        import tensorflow as tf
        tflite = tf.lite
        USE_TFLITE_RUNTIME = False
    except ImportError:
        raise ImportError("Please install either tflite-runtime or tensorflow")

# Try to import Keras for .keras model support
try:
    import keras
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("⚠️  Keras not available. Only TFLite models will be supported.")


class Config:
    """Configuration for the application"""
    # Paths
    BASE_DIR = Path(__file__).parent
    MODELS_DIR = BASE_DIR / "models"
    RESULTS_DIR = BASE_DIR / "results"
    CAPTURED_IMAGES_DIR = RESULTS_DIR / "captured_images"
    
    # Model settings
    MODEL_PATH = MODELS_DIR / "palm_classifier_float16.tflite"  # Default TFLite
    # MODEL_PATH = MODELS_DIR / "palm_classifier_final.keras"  # Alternative Keras
    
    # Image settings
    IMG_SIZE = (224, 224)
    PREVIEW_SIZE = (320, 240)
    
    # Classification settings
    CLASS_NAMES = ['sawit_mentah', 'sawit_matang']
    CONFIDENCE_THRESHOLD = 0.70
    ENTROPY_THRESHOLD = 0.7
    
    # Camera settings
    CAMERA_RESOLUTION = (640, 480)
    FRAME_RATE = 30
    
    # Results logging
    RESULTS_LOG = RESULTS_DIR / "classification_results.csv"
    
    @classmethod
    def create_dirs(cls):
        """Create necessary directories"""
        cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.CAPTURED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)


class ModelInference:
    """Handle model loading and inference"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None
        self.interpreter = None
        self.is_tflite = self.model_path.suffix == '.tflite'
        
        self.load_model()
    
    def load_model(self):
        """Load model (TFLite or Keras)"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        print(f"\n📥 Loading model from {self.model_path}")
        
        if self.is_tflite:
            self.load_tflite_model()
        else:
            self.load_keras_model()
        
        print("✅ Model loaded successfully!\n")
    
    def load_tflite_model(self):
        """Load TFLite model"""
        self.interpreter = tflite.Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"   Input shape: {self.input_details[0]['shape']}")
        print(f"   Input type: {self.input_details[0]['dtype']}")
    
    def load_keras_model(self):
        """Load Keras model"""
        if not KERAS_AVAILABLE:
            raise ImportError("Keras is not available. Please install tensorflow or keras.")
        
        self.model = keras.models.load_model(str(self.model_path))
        print(f"   Input shape: {self.model.input_shape}")
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for inference"""
        if self.is_tflite:
            img = Image.open(image_path).convert('RGB')
            img = img.resize(Config.IMG_SIZE)
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
        else:
            img = keras.preprocessing.image.load_img(
                image_path, 
                target_size=Config.IMG_SIZE
            )
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, image_path: str):
        """
        Run inference on image
        
        Returns:
            dict with keys: predictions, predicted_class, confidence, is_unknown, reason
        """
        img_array = self.preprocess_image(image_path)
        
        if self.is_tflite:
            # TFLite inference
            self.interpreter.set_tensor(self.input_details[0]['index'], img_array)
            self.interpreter.invoke()
            predictions = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        else:
            # Keras inference
            predictions = self.model.predict(img_array, verbose=0)[0]
        
        # Get predicted class
        predicted_idx = np.argmax(predictions)
        predicted_class = Config.CLASS_NAMES[predicted_idx]
        confidence = float(predictions[predicted_idx])
        
        # Calculate entropy for uncertainty detection
        entropy = self.calculate_entropy(predictions)
        
        # Determine if prediction is unknown
        is_unknown, reason = self.is_prediction_uncertain(
            predictions, confidence, entropy
        )
        
        return {
            'predictions': predictions.tolist(),
            'predicted_class': predicted_class,
            'confidence': confidence,
            'entropy': entropy,
            'is_unknown': is_unknown,
            'reason': reason
        }
    
    @staticmethod
    def calculate_entropy(probabilities: np.ndarray) -> float:
        """Calculate Shannon entropy of predictions"""
        epsilon = 1e-10
        probabilities = np.clip(probabilities, epsilon, 1.0)
        entropy = -np.sum(probabilities * np.log(probabilities))
        
        # Normalize to [0, 1]
        max_entropy = -np.log(1.0 / len(probabilities))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        return float(normalized_entropy)
    
    @staticmethod
    def is_prediction_uncertain(probabilities, confidence, entropy):
        """Check if prediction is uncertain (unknown)"""
        if confidence < Config.CONFIDENCE_THRESHOLD:
            return True, f"Low confidence ({confidence:.3f} < {Config.CONFIDENCE_THRESHOLD})"
        
        if entropy > Config.ENTROPY_THRESHOLD:
            return True, f"High uncertainty (entropy: {entropy:.3f})"
        
        return False, "Confident prediction"


class CameraApp:
    """Main Camera Application with GUI"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("🌴 Klasifikasi Kematangan Sawit - Raspberry Pi")
        self.root.geometry("700x800")  # Increased height
        self.root.resizable(True, True)  # Allow resizing
        
        # Initialize configuration
        Config.create_dirs()
        
        # Load model
        try:
            self.model_inference = ModelInference(str(Config.MODEL_PATH))
            self.model_loaded = True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            self.model_loaded = False
        
        # Initialize camera
        self.picam2 = None
        if CAMERA_AVAILABLE:
            try:
                self.init_camera()
                self.camera_active = True
            except Exception as e:
                print(f"⚠️  Camera initialization failed: {e}")
                self.camera_active = False
        else:
            self.camera_active = False
        
        # Current image path
        self.current_image_path = None
        
        # Build GUI
        self.build_gui()
        
        # Start camera update loop if camera is active
        if self.camera_active:
            self.update_camera_frame()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def init_camera(self):
        """Initialize Raspberry Pi Camera"""
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"format": "RGB888", "size": Config.CAMERA_RESOLUTION}
        )
        self.picam2.configure(config)
        self.picam2.start()
        print("✅ Camera initialized successfully")
    
    def build_gui(self):
        """Build the GUI interface with scrollbar support"""
        # Create main canvas with scrollbar
        main_canvas = tk.Canvas(self.root, highlightthickness=0)
        scrollbar = tk.Scrollbar(self.root, orient="vertical", command=main_canvas.yview)
        
        # Scrollable frame inside canvas
        scrollable_frame = tk.Frame(main_canvas, padx=10, pady=10)
        
        # Configure scroll region when frame size changes
        scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )
        
        # Create window in canvas
        canvas_frame = main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        # Update canvas window width when canvas is resized
        def on_canvas_configure(event):
            main_canvas.itemconfig(canvas_frame, width=event.width)
        
        main_canvas.bind("<Configure>", on_canvas_configure)
        main_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        main_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Enable mousewheel scrolling
        def on_mousewheel(event):
            main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def on_mousewheel_linux(event):
            if event.num == 4:
                main_canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                main_canvas.yview_scroll(1, "units")
        
        # Bind mousewheel for different platforms
        main_canvas.bind_all("<MouseWheel>", on_mousewheel)  # Windows/Mac
        main_canvas.bind_all("<Button-4>", on_mousewheel_linux)  # Linux scroll up
        main_canvas.bind_all("<Button-5>", on_mousewheel_linux)  # Linux scroll down
        
        # Now build all components in scrollable_frame instead of main_frame
        main_frame = scrollable_frame
        
        # Top section: Camera and Result preview
        preview_frame = tk.Frame(main_frame)
        preview_frame.pack(fill=tk.X, expand=False, pady=5)
        
        # Left: Camera preview
        camera_container = tk.LabelFrame(preview_frame, text="📷 Camera Live Preview", 
                                        font=("Arial", 10, "bold"))
        camera_container.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.video_label = tk.Label(camera_container, bg="black", 
                                    width=320, height=240)
        self.video_label.pack(padx=5, pady=5)
        
        if not self.camera_active:
            no_camera_label = tk.Label(camera_container, 
                                       text="Camera not available", 
                                       fg="red")
            no_camera_label.pack()
        
        # Right: Captured image preview
        result_container = tk.LabelFrame(preview_frame, text="🖼️ Captured Image", 
                                        font=("Arial", 10, "bold"))
        result_container.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.captured_label = tk.Label(result_container, bg="gray", 
                                       width=320, height=240)
        self.captured_label.pack(padx=5, pady=5)
        
        # Control buttons
        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill=tk.X, expand=False, pady=10)
        
        btn_style = {"font": ("Arial", 10), "padx": 10, "pady": 5, "width": 18}
        
        if self.camera_active:
            self.capture_btn = tk.Button(control_frame, text="📸 Capture Camera",
                                         command=self.capture_image,
                                         bg="#4CAF50", fg="white", **btn_style)
            self.capture_btn.grid(row=0, column=0, padx=5, pady=5)
        
        self.select_btn = tk.Button(control_frame, text="📂 Select File",
                                    command=self.select_image,
                                    bg="#2196F3", fg="white", **btn_style)
        self.select_btn.grid(row=0, column=1, padx=5, pady=5)
        
        self.clear_btn = tk.Button(control_frame, text="🗑️ Clear",
                                   command=self.clear_results,
                                   bg="#f44336", fg="white", **btn_style)
        self.clear_btn.grid(row=0, column=2, padx=5, pady=5)
        
        # Results display
        results_frame = tk.LabelFrame(main_frame, text="📊 Classification Results", 
                                     font=("Arial", 10, "bold"), padx=10, pady=10)
        results_frame.pack(fill=tk.BOTH, expand=False, pady=10)
        
        # File path
        file_frame = tk.Frame(results_frame)
        file_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(file_frame, text="File:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        self.file_label = tk.Label(file_frame, text="-", font=("Arial", 9), 
                                   fg="blue", wraplength=500, justify=tk.LEFT)
        self.file_label.pack(side=tk.LEFT, padx=10)
        
        # Prediction result
        result_frame = tk.Frame(results_frame)
        result_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(result_frame, text="Result:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        self.result_label = tk.Label(result_frame, text="-", 
                                     font=("Arial", 20, "bold"), fg="gray")
        self.result_label.pack(side=tk.LEFT, padx=10)
        
        # Confidence
        conf_frame = tk.Frame(results_frame)
        conf_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(conf_frame, text="Confidence:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        self.confidence_label = tk.Label(conf_frame, text="-", font=("Arial", 12))
        self.confidence_label.pack(side=tk.LEFT, padx=10)
        
        # All predictions
        prob_frame = tk.Frame(results_frame)
        prob_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(prob_frame, text="Probabilities:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.prob_text = tk.Text(prob_frame, height=3, font=("Courier", 9))
        self.prob_text.pack(fill=tk.X, pady=5)
        self.prob_text.config(state=tk.DISABLED)
        
        # Status bar
        self.status_label = tk.Label(main_frame, text="Ready", 
                                     font=("Arial", 9), fg="green",
                                     relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(fill=tk.X, side=tk.BOTTOM)
    
    def update_camera_frame(self):
        """Update camera preview frame"""
        if not self.camera_active or self.picam2 is None:
            return
        
        try:
            frame = self.picam2.capture_array()
            img_pil = Image.fromarray(frame)
            img_pil = img_pil.resize(Config.PREVIEW_SIZE)
            img_tk = ImageTk.PhotoImage(img_pil)
            
            self.video_label.imgtk = img_tk
            self.video_label.configure(image=img_tk)
        except Exception as e:
            print(f"Camera frame update error: {e}")
        
        # Schedule next update
        self.root.after(33, self.update_camera_frame)  # ~30 FPS
    
    def capture_image(self):
        """Capture image from camera"""
        if not self.camera_active:
            messagebox.showwarning("Warning", "Camera is not available!")
            return
        
        if not self.model_loaded:
            messagebox.showwarning("Warning", "Model is not loaded!")
            return
        
        try:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"
            self.current_image_path = Config.CAPTURED_IMAGES_DIR / filename
            
            # Capture and save image
            self.picam2.capture_file(str(self.current_image_path))
            self.status_label.config(text=f"Image captured: {filename}", fg="blue")
            
            # Display captured image
            self.display_captured_image()
            
            # Run inference
            self.run_inference()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to capture image:\n{str(e)}")
            self.status_label.config(text=f"Capture failed: {str(e)}", fg="red")
    
    def select_image(self):
        """Select image from file system"""
        if not self.model_loaded:
            messagebox.showwarning("Warning", "Model is not loaded!")
            return
        
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            self.current_image_path = Path(file_path)
            self.status_label.config(text=f"Image selected: {self.current_image_path.name}", 
                                    fg="blue")
            
            # Display selected image
            self.display_captured_image()
            
            # Run inference
            self.run_inference()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
            self.status_label.config(text=f"Load failed: {str(e)}", fg="red")
    
    def display_captured_image(self):
        """Display captured/selected image"""
        if self.current_image_path is None or not self.current_image_path.exists():
            return
        
        img_pil = Image.open(self.current_image_path)
        img_pil = img_pil.resize(Config.PREVIEW_SIZE)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        self.captured_label.configure(image=img_tk)
        self.captured_label.image = img_tk
    
    def run_inference(self):
        """Run model inference on current image"""
        if self.current_image_path is None:
            return
        
        try:
            self.status_label.config(text="Running inference...", fg="orange")
            self.root.update()
            
            # Run prediction
            start_time = time.time()
            result = self.model_inference.predict(str(self.current_image_path))
            inference_time = (time.time() - start_time) * 1000  # ms
            
            # Update UI with results
            self.display_results(result, inference_time)
            
            # Log results
            self.log_results(result, inference_time)
            
            self.status_label.config(
                text=f"Inference completed in {inference_time:.1f}ms", 
                fg="green"
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Inference failed:\n{str(e)}")
            self.status_label.config(text=f"Inference failed: {str(e)}", fg="red")
    
    def display_results(self, result, inference_time):
        """Display classification results in GUI"""
        # Update file path
        self.file_label.config(text=str(self.current_image_path))
        
        # Update prediction result
        if result['is_unknown']:
            self.result_label.config(
                text="❓ UNKNOWN", 
                fg="orange"
            )
            result_text = f"UNKNOWN (Top: {result['predicted_class']})"
        else:
            class_display = result['predicted_class'].replace('_', ' ').title()
            self.result_label.config(
                text=f"✅ {class_display}", 
                fg="green" if result['predicted_class'] == 'sawit_matang' else "red"
            )
            result_text = class_display
        
        # Update confidence
        conf_text = f"{result['confidence']:.4f} ({result['confidence']*100:.2f}%)"
        self.confidence_label.config(text=conf_text)
        
        # Update probabilities
        self.prob_text.config(state=tk.NORMAL)
        self.prob_text.delete(1.0, tk.END)
        
        prob_text = ""
        for i, class_name in enumerate(Config.CLASS_NAMES):
            prob = result['predictions'][i]
            bar = "█" * int(prob * 30)
            prob_text += f"{class_name:15s}: {prob:.4f} {bar}\n"
        
        prob_text += f"\nEntropy: {result['entropy']:.4f}\n"
        prob_text += f"Inference time: {inference_time:.1f}ms"
        
        self.prob_text.insert(1.0, prob_text)
        self.prob_text.config(state=tk.DISABLED)
    
    def log_results(self, result, inference_time):
        """Log classification results to CSV file"""
        try:
            # Create CSV file with headers if it doesn't exist
            if not Config.RESULTS_LOG.exists():
                with open(Config.RESULTS_LOG, 'w') as f:
                    f.write("timestamp,image_path,predicted_class,confidence,is_unknown,"
                           "entropy,inference_time_ms,sawit_mentah_prob,sawit_matang_prob\n")
            
            # Append results
            with open(Config.RESULTS_LOG, 'a') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                line = (
                    f"{timestamp},"
                    f"{self.current_image_path},"
                    f"{result['predicted_class']},"
                    f"{result['confidence']:.4f},"
                    f"{result['is_unknown']},"
                    f"{result['entropy']:.4f},"
                    f"{inference_time:.2f},"
                    f"{result['predictions'][0]:.4f},"
                    f"{result['predictions'][1]:.4f}\n"
                )
                f.write(line)
            
            print(f"✅ Results logged to {Config.RESULTS_LOG}")
            
        except Exception as e:
            print(f"⚠️  Failed to log results: {e}")
    
    def clear_results(self):
        """Clear all results from UI"""
        self.current_image_path = None
        self.file_label.config(text="-")
        self.result_label.config(text="-", fg="gray")
        self.confidence_label.config(text="-")
        
        self.prob_text.config(state=tk.NORMAL)
        self.prob_text.delete(1.0, tk.END)
        self.prob_text.config(state=tk.DISABLED)
        
        # Clear captured image
        self.captured_label.configure(image='')
        self.captured_label.image = None
        
        self.status_label.config(text="Cleared", fg="blue")
    
    def on_close(self):
        """Handle application close"""
        if self.picam2 is not None:
            try:
                self.picam2.stop()
                self.picam2.close()
                print("✅ Camera closed successfully")
            except:
                pass
        
        self.root.destroy()


def main():
    """Main application entry point"""
    print("="*60)
    print("🌴 PALM FRUIT CLASSIFICATION - RASPBERRY PI")
    print("="*60)
    print(f"Model: {Config.MODEL_PATH}")
    print(f"Classes: {Config.CLASS_NAMES}")
    print(f"Confidence threshold: {Config.CONFIDENCE_THRESHOLD}")
    print("="*60 + "\n")
    
    # Check if model exists
    if not Config.MODEL_PATH.exists():
        print(f"❌ Model not found at {Config.MODEL_PATH}")
        print("\nPlease copy your model file to the models directory:")
        print(f"   {Config.MODELS_DIR}/")
        print("\nSupported formats:")
        print("   - palm_classifier_float16.tflite (recommended)")
        print("   - palm_classifier_final.keras")
        return
    
    # Create and run application
    root = tk.Tk()
    app = CameraApp(root)  # Must keep reference to prevent garbage collection
    
    # Store app reference in root to keep it alive
    root.app_instance = app
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
        app.on_close()
    except Exception as e:
        print(f"\n❌ Application error: {e}")
        if hasattr(app, 'on_close'):
            app.on_close()


if __name__ == "__main__":
    main()