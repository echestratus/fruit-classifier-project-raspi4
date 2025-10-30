#!/usr/bin/env python3
"""
GUI Debug Script - Test Tkinter and identify issues
"""
import tkinter as tk
from tkinter import messagebox

def test_basic_window():
    """Test basic window creation"""
    print("\n1️⃣  Testing basic window...")
    root = tk.Tk()
    root.title("Test Window")
    root.geometry("400x300")
    
    label = tk.Label(root, text="✅ If you see this, Tkinter works!", 
                    font=("Arial", 14))
    label.pack(pady=20)
    
    def close():
        print("   ✅ Basic window test passed")
        root.destroy()
    
    btn = tk.Button(root, text="Close", command=close, 
                   font=("Arial", 12), bg="red", fg="white")
    btn.pack(pady=10)
    
    root.after(100, lambda: print("   Window displayed"))
    root.mainloop()

def test_frames():
    """Test frame layout"""
    print("\n2️⃣  Testing frames and layout...")
    root = tk.Tk()
    root.title("Frame Test")
    root.geometry("500x400")
    
    # Top frame
    top_frame = tk.Frame(root, bg="lightblue", height=100)
    top_frame.pack(fill=tk.X, padx=10, pady=5)
    tk.Label(top_frame, text="Top Frame", bg="lightblue").pack()
    
    # Middle frame
    mid_frame = tk.Frame(root, bg="lightgreen", height=100)
    mid_frame.pack(fill=tk.X, padx=10, pady=5)
    tk.Label(mid_frame, text="Middle Frame", bg="lightgreen").pack()
    
    # Button frame
    btn_frame = tk.Frame(root, bg="lightyellow")
    btn_frame.pack(fill=tk.X, padx=10, pady=5)
    
    tk.Button(btn_frame, text="Button 1").grid(row=0, column=0, padx=5)
    tk.Button(btn_frame, text="Button 2").grid(row=0, column=1, padx=5)
    tk.Button(btn_frame, text="Button 3").grid(row=0, column=2, padx=5)
    
    # Bottom frame
    bottom_frame = tk.Frame(root, bg="lightcoral", height=100)
    bottom_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    tk.Label(bottom_frame, text="Bottom Frame (expandable)", 
            bg="lightcoral").pack()
    
    def close():
        print("   ✅ Frame test passed")
        root.destroy()
    
    tk.Button(root, text="Close", command=close).pack(pady=10)
    
    root.mainloop()

def test_pack_vs_grid():
    """Test pack vs grid layout"""
    print("\n3️⃣  Testing pack vs grid...")
    root = tk.Tk()
    root.title("Pack vs Grid")
    root.geometry("600x400")
    
    # Pack example
    pack_frame = tk.LabelFrame(root, text="PACK Layout", font=("Arial", 12, "bold"))
    pack_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    tk.Button(pack_frame, text="Pack Button 1").pack(pady=5)
    tk.Button(pack_frame, text="Pack Button 2").pack(pady=5)
    tk.Button(pack_frame, text="Pack Button 3").pack(pady=5)
    
    # Grid example
    grid_frame = tk.LabelFrame(root, text="GRID Layout", font=("Arial", 12, "bold"))
    grid_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    tk.Button(grid_frame, text="Grid Button 1").grid(row=0, column=0, pady=5, padx=5)
    tk.Button(grid_frame, text="Grid Button 2").grid(row=0, column=1, pady=5, padx=5)
    tk.Button(grid_frame, text="Grid Button 3").grid(row=1, column=0, pady=5, padx=5)
    tk.Button(grid_frame, text="Grid Button 4").grid(row=1, column=1, pady=5, padx=5)
    
    def close():
        print("   ✅ Layout test passed")
        root.destroy()
    
    tk.Button(root, text="Close", command=close).pack(side=tk.BOTTOM, pady=10)
    
    root.mainloop()

def test_scrollbar():
    """Test scrollbar"""
    print("\n4️⃣  Testing scrollbar...")
    root = tk.Tk()
    root.title("Scrollbar Test")
    root.geometry("400x300")
    
    # Create frame with scrollbar
    canvas = tk.Canvas(root)
    scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    # Add many items
    for i in range(50):
        tk.Label(scrollable_frame, text=f"Item {i+1}").pack()
    
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    def close():
        print("   ✅ Scrollbar test passed")
        root.destroy()
    
    tk.Button(root, text="Close", command=close).pack(side=tk.BOTTOM)
    
    root.mainloop()

def test_geometry():
    """Test window geometry"""
    print("\n5️⃣  Testing window geometry...")
    root = tk.Tk()
    root.title("Geometry Test")
    
    sizes = [
        ("Small (400x300)", "400x300"),
        ("Medium (600x500)", "600x500"),
        ("Large (800x700)", "800x700"),
        ("Extra Large (1000x900)", "1000x900")
    ]
    
    current_size = [0]
    
    def change_size():
        current_size[0] = (current_size[0] + 1) % len(sizes)
        name, geom = sizes[current_size[0]]
        root.geometry(geom)
        label.config(text=f"Current: {name}\n{geom}")
        print(f"   Changed to: {geom}")
    
    label = tk.Label(root, text=f"Current: {sizes[0][0]}\n{sizes[0][1]}", 
                    font=("Arial", 14))
    label.pack(pady=20)
    
    tk.Button(root, text="Change Size", command=change_size,
             font=("Arial", 12)).pack(pady=10)
    
    def close():
        print("   ✅ Geometry test passed")
        root.destroy()
    
    tk.Button(root, text="Close", command=close).pack(pady=10)
    
    root.geometry(sizes[0][1])
    root.mainloop()

def test_full_layout():
    """Test full application layout"""
    print("\n6️⃣  Testing full application layout...")
    root = tk.Tk()
    root.title("Full Layout Test")
    root.geometry("800x900")
    
    # Title
    tk.Label(root, text="🌴 Full Application Layout Test", 
            font=("Arial", 16, "bold")).pack(pady=10)
    
    # Preview section
    preview_frame = tk.Frame(root)
    preview_frame.pack(fill=tk.X, pady=5)
    
    cam_frame = tk.LabelFrame(preview_frame, text="Camera Preview")
    cam_frame.pack(side=tk.LEFT, padx=10)
    tk.Label(cam_frame, text="Camera\nPreview\nArea", 
            width=30, height=10, bg="black", fg="white").pack(padx=10, pady=10)
    
    img_frame = tk.LabelFrame(preview_frame, text="Captured Image")
    img_frame.pack(side=tk.LEFT, padx=10)
    tk.Label(img_frame, text="Captured\nImage\nArea", 
            width=30, height=10, bg="gray", fg="white").pack(padx=10, pady=10)
    
    # Buttons
    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=10)
    
    tk.Button(btn_frame, text="📸 Capture", bg="green", fg="white",
             width=15, height=2).grid(row=0, column=0, padx=5)
    tk.Button(btn_frame, text="📂 Open", bg="blue", fg="white",
             width=15, height=2).grid(row=0, column=1, padx=5)
    tk.Button(btn_frame, text="🗑️ Clear", bg="red", fg="white",
             width=15, height=2).grid(row=0, column=2, padx=5)
    
    # Results
    result_frame = tk.LabelFrame(root, text="📊 Results", font=("Arial", 12, "bold"))
    result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    tk.Label(result_frame, text="Prediction: Sawit Matang", 
            font=("Arial", 18, "bold"), fg="green").pack(pady=10)
    tk.Label(result_frame, text="Confidence: 95.34%",
            font=("Arial", 14)).pack(pady=5)
    
    prob_text = tk.Text(result_frame, height=5, width=50)
    prob_text.pack(pady=10)
    prob_text.insert(1.0, "sawit_mentah   : 0.0466\nsawit_matang   : 0.9534")
    
    # Status
    tk.Label(root, text="Status: Ready", relief=tk.SUNKEN, 
            anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)
    
    def close():
        print("   ✅ Full layout test passed")
        root.destroy()
    
    tk.Button(root, text="Close Test", command=close, bg="orange",
             font=("Arial", 12)).pack(side=tk.BOTTOM, pady=5)
    
    print("   All components should be visible")
    print("   If buttons are missing, there's a layout issue")
    
    root.mainloop()

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🐛 GUI DEBUGGING TESTS")
    print("="*60)
    print("\nThis will run several tests to identify GUI issues.")
    print("Close each window to proceed to the next test.\n")
    
    tests = [
        ("Basic Window", test_basic_window),
        ("Frames", test_frames),
        ("Pack vs Grid", test_pack_vs_grid),
        ("Scrollbar", test_scrollbar),
        ("Geometry", test_geometry),
        ("Full Layout", test_full_layout),
    ]
    
    for i, (name, test_func) in enumerate(tests, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}/{len(tests)}: {name}")
        print(f"{'='*60}")
        
        try:
            test_func()
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            break
    
    print("\n" + "="*60)
    print("✅ All GUI tests completed!")
    print("="*60)
    print("\nIf all tests passed but your app still has issues:")
    print("1. Try raspberry_pi_simple_app.py instead")
    print("2. Check window size: root.geometry('800x900')")
    print("3. Run on RPi desktop (not SSH without X11)")
    print("4. Update tkinter: sudo apt-get install python3-tk")
    print("\n")

if __name__ == "__main__":
    main()