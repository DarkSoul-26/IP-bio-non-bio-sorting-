import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import threading
import time
import numpy as np
import os
import tensorflow as tf

class BiodegradableDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Biodegradable Item Detector")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f5f5f5")
        
        # Set app icon and theme
        self.set_theme()
        
        # Variables
        self.webcam_active = False
        self.detection_active = False
        self.webcam_thread = None
        self.cap = None
        self.current_frame = None
        self.detection_count = {"Biodegradable": 0, "Non-biodegradable": 0}
        self.last_detection_time = time.time()
        self.detection_cooldown = 1.0  # seconds between detections
        self.model = None
        self.model_loaded = False
        
        # Create UI components
        self.create_header()
        self.create_main_content()
        self.create_footer()
        
        # Load TensorFlow model
        self.load_model()
        
        # Bind close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def load_model(self):
        try:
            # Show loading message
            self.show_status("Loading model... Please wait")
            
            # Load the model in a separate thread to prevent UI freezing
            threading.Thread(target=self._load_model_thread, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Model Loading Error", f"Failed to load model: {str(e)}")
            self.show_status("Model loading failed")
    
    def _load_model_thread(self):
        try:
            # Load the TensorFlow SavedModel
            self.model = tf.saved_model.load('model.savedmodel')
            self.model_loaded = True
            
            # Update UI from main thread
            self.root.after(0, lambda: self.show_status("Model loaded successfully"))
            self.root.after(0, lambda: self.info_text.config(
                text="Model loaded successfully. Start the webcam and detection to begin classifying items."
            ))
        except Exception as e:
            # Update UI from main thread
            self.root.after(0, lambda: messagebox.showerror("Model Loading Error", f"Failed to load model: {str(e)}"))
            self.root.after(0, lambda: self.show_status("Model loading failed"))
    
    def set_theme(self):
        # Configure ttk styles
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure button styles
        self.style.configure('Primary.TButton', 
                            background='#4CAF50', 
                            foreground='white', 
                            font=('Helvetica', 12, 'bold'),
                            padding=10)
        
        self.style.configure('Secondary.TButton', 
                            background='#2196F3', 
                            foreground='white', 
                            font=('Helvetica', 12),
                            padding=10)
        
        self.style.configure('Danger.TButton', 
                            background='#f44336', 
                            foreground='white', 
                            font=('Helvetica', 12),
                            padding=10)
        
        # Configure progress bar
        self.style.configure("green.Horizontal.TProgressbar", 
                            background='#4CAF50',
                            troughcolor='#e0e0e0')
        
        self.style.configure("red.Horizontal.TProgressbar", 
                            background='#f44336',
                            troughcolor='#e0e0e0')
    
    def create_header(self):
        # Create gradient header
        header_frame = tk.Frame(self.root, height=80, bg="#388E3C")
        header_frame.pack(fill=tk.X)
        
        # App title
        title_label = tk.Label(
            header_frame, 
            text="Biodegradable Item Detector", 
            font=("Helvetica", 24, "bold"),
            bg="#388E3C",
            fg="white"
        )
        title_label.pack(pady=20)
        
        # Subtitle
        subtitle_label = tk.Label(
            header_frame, 
            text="Real-time detection using webcam", 
            font=("Helvetica", 12),
            bg="#388E3C",
            fg="#E8F5E9"
        )
        subtitle_label.place(x=20, y=55)
    
    def create_main_content(self):
        main_frame = tk.Frame(self.root, bg="#f5f5f5")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel - Video feed
        self.video_frame = tk.Frame(main_frame, bg="#ffffff", width=640, height=480, 
                                   relief=tk.RIDGE, borderwidth=1)
        self.video_frame.pack(side=tk.LEFT, padx=(0, 10))
        self.video_frame.pack_propagate(False)
        
        self.video_label = tk.Label(self.video_frame, bg="#f0f0f0")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Placeholder text when no webcam is active
        self.placeholder_label = tk.Label(
            self.video_label,
            text="Webcam feed will appear here",
            font=("Helvetica", 14),
            bg="#f0f0f0"
        )
        self.placeholder_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # Right panel - Controls and results
        right_panel = tk.Frame(main_frame, bg="#f5f5f5", width=300)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Controls section
        controls_frame = tk.LabelFrame(right_panel, text="Controls", bg="#f5f5f5", 
                                      font=("Helvetica", 12, "bold"), padx=10, pady=10)
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Webcam controls
        self.webcam_btn = ttk.Button(
            controls_frame,
            text="Start Webcam",
            command=self.toggle_webcam,
            style='Secondary.TButton'
        )
        self.webcam_btn.pack(fill=tk.X, pady=5)
        
        # Detection controls
        self.detection_btn = ttk.Button(
            controls_frame,
            text="Start Detection",
            command=self.toggle_detection,
            style='Primary.TButton',
            state=tk.DISABLED
        )
        self.detection_btn.pack(fill=tk.X, pady=5)
        
        # Capture frame button
        self.capture_btn = ttk.Button(
            controls_frame,
            text="Capture Frame",
            command=self.capture_frame,
            style='Secondary.TButton',
            state=tk.DISABLED
        )
        self.capture_btn.pack(fill=tk.X, pady=5)
        
        # Reset button
        reset_btn = ttk.Button(
            controls_frame,
            text="Reset Statistics",
            command=self.reset_stats,
            style='Danger.TButton'
        )
        reset_btn.pack(fill=tk.X, pady=5)
        
        # Webcam selection (dropdown)
        camera_frame = tk.Frame(controls_frame, bg="#f5f5f5")
        camera_frame.pack(fill=tk.X, pady=5)
        
        camera_label = tk.Label(camera_frame, text="Camera:", bg="#f5f5f5", font=("Helvetica", 10))
        camera_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.camera_var = tk.StringVar(value="0")
        camera_options = ["0", "1", "2", "3"]  # Camera indices
        camera_dropdown = ttk.Combobox(camera_frame, textvariable=self.camera_var, values=camera_options, width=5)
        camera_dropdown.pack(side=tk.LEFT)
        
        # Model settings
        model_frame = tk.Frame(controls_frame, bg="#f5f5f5")
        model_frame.pack(fill=tk.X, pady=5)
        
        # Confidence threshold slider
        threshold_label = tk.Label(model_frame, text="Detection Threshold:", bg="#f5f5f5", font=("Helvetica", 10))
        threshold_label.pack(anchor=tk.W)
        
        self.threshold_var = tk.DoubleVar(value=0.5)
        threshold_slider = ttk.Scale(
            model_frame, 
            from_=0.1, 
            to=0.9, 
            orient=tk.HORIZONTAL, 
            variable=self.threshold_var,
            length=200
        )
        threshold_slider.pack(fill=tk.X)
        
        threshold_value_frame = tk.Frame(model_frame, bg="#f5f5f5")
        threshold_value_frame.pack(fill=tk.X)
        
        tk.Label(threshold_value_frame, text="Low", bg="#f5f5f5", font=("Helvetica", 8)).pack(side=tk.LEFT)
        
        self.threshold_value_label = tk.Label(
            threshold_value_frame, 
            textvariable=tk.StringVar(value=f"{self.threshold_var.get():.1f}"),
            bg="#f5f5f5", 
            font=("Helvetica", 8, "bold")
        )
        self.threshold_value_label.pack(side=tk.LEFT, padx=70)
        
        tk.Label(threshold_value_frame, text="High", bg="#f5f5f5", font=("Helvetica", 8)).pack(side=tk.RIGHT)
        
        # Update threshold value label when slider changes
        threshold_slider.bind("<Motion>", lambda e: self.threshold_value_label.config(
            text=f"{self.threshold_var.get():.1f}"
        ))
        
        # Results section
        results_frame = tk.LabelFrame(right_panel, text="Detection Results", bg="#f5f5f5", 
                                     font=("Helvetica", 12, "bold"), padx=10, pady=10)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Current detection result
        self.result_label = tk.Label(
            results_frame,
            text="No detection yet",
            font=("Helvetica", 14),
            bg="#f5f5f5"
        )
        self.result_label.pack(pady=10)
        
        # Confidence level
        confidence_frame = tk.Frame(results_frame, bg="#f5f5f5")
        confidence_frame.pack(fill=tk.X, pady=5)
        
        confidence_label = tk.Label(
            confidence_frame,
            text="Confidence:",
            font=("Helvetica", 10),
            bg="#f5f5f5"
        )
        confidence_label.pack(anchor=tk.W)
        
        self.confidence_bar = ttk.Progressbar(
            confidence_frame,
            orient=tk.HORIZONTAL,
            length=200,
            mode='determinate',
            style="green.Horizontal.TProgressbar"
        )
        self.confidence_bar.pack(fill=tk.X, pady=5)
        
        self.confidence_value = tk.Label(
            confidence_frame,
            text="0%",
            font=("Helvetica", 10),
            bg="#f5f5f5"
        )
        self.confidence_value.pack(anchor=tk.E)
        
        # Statistics
        stats_frame = tk.LabelFrame(results_frame, text="Statistics", bg="#f5f5f5", padx=5, pady=5)
        stats_frame.pack(fill=tk.X, pady=10)
        
        # Biodegradable count
        bio_frame = tk.Frame(stats_frame, bg="#f5f5f5")
        bio_frame.pack(fill=tk.X, pady=2)
        
        bio_label = tk.Label(
            bio_frame,
            text="Biodegradable:",
            font=("Helvetica", 10),
            bg="#f5f5f5"
        )
        bio_label.pack(side=tk.LEFT)
        
        self.bio_count = tk.Label(
            bio_frame,
            text="0",
            font=("Helvetica", 10, "bold"),
            bg="#f5f5f5",
            fg="#4CAF50"
        )
        self.bio_count.pack(side=tk.RIGHT)
        
        # Non-biodegradable count
        nonbio_frame = tk.Frame(stats_frame, bg="#f5f5f5")
        nonbio_frame.pack(fill=tk.X, pady=2)
        
        nonbio_label = tk.Label(
            nonbio_frame,
            text="Non-biodegradable:",
            font=("Helvetica", 10),
            bg="#f5f5f5"
        )
        nonbio_label.pack(side=tk.LEFT)
        
        self.nonbio_count = tk.Label(
            nonbio_frame,
            text="0",
            font=("Helvetica", 10, "bold"),
            bg="#f5f5f5",
            fg="#f44336"
        )
        self.nonbio_count.pack(side=tk.RIGHT)
        
        # Information box
        info_frame = tk.Frame(results_frame, bg="#e8f5e9", padx=10, pady=10, relief=tk.GROOVE, bd=1)
        info_frame.pack(fill=tk.X, pady=10)
        
        self.info_text = tk.Label(
            info_frame,
            text="Loading model... Please wait.",
            font=("Helvetica", 10),
            bg="#e8f5e9",
            wraplength=250,
            justify=tk.LEFT
        )
        self.info_text.pack(fill=tk.X)
    
    def create_footer(self):
        footer_frame = tk.Frame(self.root, height=30, bg="#e0e0e0")
        footer_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_label = tk.Label(
            footer_frame,
            text="Loading model...",
            font=("Helvetica", 9),
            bg="#e0e0e0"
        )
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        self.status_text = tk.StringVar(value="Webcam: Inactive | Detection: Inactive")
        status_info = tk.Label(
            footer_frame,
            textvariable=self.status_text,
            font=("Helvetica", 9),
            bg="#e0e0e0"
        )
        status_info.pack(side=tk.RIGHT, padx=10)
    
    def show_status(self, message):
        self.status_label.config(text=message)
    
    def toggle_webcam(self):
        if not self.webcam_active:
            self.start_webcam()
        else:
            self.stop_webcam()
    
    def start_webcam(self):
        if not self.model_loaded:
            messagebox.showwarning("Model Not Loaded", "Please wait for the model to finish loading.")
            return
            
        try:
            camera_index = int(self.camera_var.get())
            self.cap = cv2.VideoCapture(camera_index)
            
            if not self.cap.isOpened():
                messagebox.showerror("Error", f"Could not open camera {camera_index}")
                return
            
            self.webcam_active = True
            self.webcam_btn.config(text="Stop Webcam")
            self.detection_btn.config(state=tk.NORMAL)
            self.capture_btn.config(state=tk.NORMAL)
            self.placeholder_label.place_forget()
            
            # Update status
            self.update_status()
            self.show_status("Webcam started")
            
            # Start webcam thread
            self.webcam_thread = threading.Thread(target=self.update_webcam, daemon=True)
            self.webcam_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start webcam: {str(e)}")
    
    def stop_webcam(self):
        self.webcam_active = False
        self.stop_detection()
        
        if self.cap and self.cap.isOpened():
            self.cap.release()
        
        self.webcam_btn.config(text="Start Webcam")
        self.detection_btn.config(state=tk.DISABLED)
        self.capture_btn.config(state=tk.DISABLED)
        
        # Show placeholder
        self.placeholder_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # Clear video display
        self.video_label.config(image="")
        
        # Update status
        self.update_status()
        self.show_status("Webcam stopped")
    
    def update_webcam(self):
        while self.webcam_active:
            ret, frame = self.cap.read()
            
            if not ret:
                self.webcam_active = False
                messagebox.showerror("Error", "Failed to capture frame from webcam")
                break
            
            # Flip the frame horizontally for a more natural view
            frame = cv2.flip(frame, 1)
            
            # Store current frame for detection
            self.current_frame = frame.copy()
            
            # If detection is active, perform detection
            if self.detection_active and time.time() - self.last_detection_time > self.detection_cooldown:
                self.perform_detection(frame)
                self.last_detection_time = time.time()
            
            # Convert to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Add detection status overlay
            status_text = "Detection: ACTIVE" if self.detection_active else "Detection: INACTIVE"
            color = (0, 255, 0) if self.detection_active else (255, 0, 0)
            cv2.putText(frame_rgb, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, color, 2, cv2.LINE_AA)
            
            # Resize to fit the display area
            frame_rgb = cv2.resize(frame_rgb, (640, 480))
            
            # Convert to ImageTk format
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Update the video label
            self.video_label.config(image=imgtk)
            self.video_label.image = imgtk
    
    def toggle_detection(self):
        if not self.detection_active:
            self.start_detection()
        else:
            self.stop_detection()
    
    def start_detection(self):
        if not self.model_loaded:
            messagebox.showwarning("Model Not Loaded", "Please wait for the model to finish loading.")
            return
            
        self.detection_active = True
        self.detection_btn.config(text="Stop Detection")
        self.info_text.config(text="Detection is active. Show items to the camera for classification.")
        self.update_status()
        self.show_status("Detection started")
    
    def stop_detection(self):
        self.detection_active = False
        self.detection_btn.config(text="Start Detection")
        self.info_text.config(text="Detection is paused. Press 'Start Detection' to resume.")
        self.update_status()
        self.show_status("Detection stopped")
    
    def capture_frame(self):
        if not self.model_loaded:
            messagebox.showwarning("Model Not Loaded", "Please wait for the model to finish loading.")
            return
            
        if self.current_frame is not None:
            self.show_status("Processing frame...")
            self.perform_detection(self.current_frame)
            self.show_status("Frame processed")
    
    def preprocess_image(self, frame):
        """Preprocess the image for model input"""
        try:
            # Resize the image to the model's expected input dimensions
            # Note: Adjust these dimensions to match your model's requirements
            input_size = (224, 224)  # Common input size for many models
            resized = cv2.resize(frame, input_size)
            
            # Convert to RGB if the model expects RGB input
            if frame.shape[2] == 3:
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Normalize pixel values to [0, 1]
            normalized = resized.astype(np.float32) / 255.0
            
            # Add batch dimension
            batched = np.expand_dims(normalized, axis=0)
            
            return batched
            
        except Exception as e:
            messagebox.showerror("Preprocessing Error", f"Error preprocessing image: {str(e)}")
            return None
    
    def perform_detection(self, frame):
        """Perform detection using the TensorFlow model"""
        try:
            # Preprocess the image
            processed_image = self.preprocess_image(frame)
            
            if processed_image is None:
                return
            
            # Convert to TensorFlow tensor
            input_tensor = tf.convert_to_tensor(processed_image)
            
            # Get the serving signature from the model
            # Note: Adjust this based on your model's signature
            serving_fn = self.model.signatures['serving_default']
            
            # Run inference
            output = serving_fn(input_tensor)
            
            # Process the output based on your model's output format
            # This is a generic example - adjust based on your model's specific output
            
            # Get the output tensor (adjust the key based on your model's output)
            # Common output keys: 'logits', 'predictions', 'output', etc.
            output_key = list(output.keys())[0]  # Get the first output key
            predictions = output[output_key].numpy()
            
            # For binary classification
            if predictions.shape[-1] == 1 or len(predictions.shape) == 1:
                # Single output neuron (binary classification)
                confidence = float(predictions[0][0]) if len(predictions.shape) > 1 else float(predictions[0])
                
                # Apply threshold
                threshold = self.threshold_var.get()
                result = "Biodegradable" if confidence >= threshold else "Non-biodegradable"
                
                # Ensure confidence is between 0 and 1
                confidence = max(0, min(1, confidence))
                
            # For multi-class classification
            else:
                # Multiple output neurons (class probabilities)
                class_idx = np.argmax(predictions[0])
                confidence = float(predictions[0][class_idx])
                
                # Map class index to label (adjust based on your model's classes)
                result = "Biodegradable" if class_idx == 0 else "Non-biodegradable"
            
            # Update detection count
            self.detection_count[result] += 1
            
            # Update UI with results
            self.update_result_display(result, confidence)
            
        except Exception as e:
            messagebox.showerror("Detection Error", f"Error during detection: {str(e)}")
            self.show_status(f"Detection error: {str(e)}")
    
    def update_result_display(self, result, confidence):
        # Update result label
        self.result_label.config(
            text=result,
            fg="#4CAF50" if result == "Biodegradable" else "#f44336"
        )
        
        # Update confidence bar
        self.confidence_bar.config(
            style="green.Horizontal.TProgressbar" if result == "Biodegradable" else "red.Horizontal.TProgressbar"
        )
        self.confidence_bar["value"] = confidence * 100
        self.confidence_value.config(text=f"{confidence:.1%}")
        
        # Update statistics
        self.bio_count.config(text=str(self.detection_count["Biodegradable"]))
        self.nonbio_count.config(text=str(self.detection_count["Non-biodegradable"]))
        
        # Update info text
        if result == "Biodegradable":
            info = "This item will decompose naturally in the environment. It can be composted or disposed of in organic waste."
        else:
            info = "This item will not decompose naturally. It should be recycled properly according to local guidelines."
        
        self.info_text.config(text=info)
    
    def reset_stats(self):
        self.detection_count = {"Biodegradable": 0, "Non-biodegradable": 0}
        self.bio_count.config(text="0")
        self.nonbio_count.config(text="0")
        self.result_label.config(text="No detection yet", fg="black")
        self.confidence_bar["value"] = 0
        self.confidence_value.config(text="0%")
        self.info_text.config(text="Statistics have been reset.")
        self.show_status("Statistics reset")
    
    def update_status(self):
        webcam_status = "Active" if self.webcam_active else "Inactive"
        detection_status = "Active" if self.detection_active else "Inactive"
        self.status_text.set(f"Webcam: {webcam_status} | Detection: {detection_status}")
    
    def on_closing(self):
        if self.webcam_active:
            self.stop_webcam()
        self.root.destroy()

# Main application
if __name__ == "__main__":
    # Enable memory growth for GPU if available
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    root = tk.Tk()
    app = BiodegradableDetectorGUI(root)
    root.mainloop()

print("TensorFlow model integration complete!")