"""
Quick Start Script for Disease Gene Detection System
Run this to get started quickly with your datasets
"""
import os
import sys

print("=" * 70)
print("🧬 DISEASE GENE DETECTION - QUICK START")
print("=" * 70)

def main():
    print("\nWelcome! This script will help you get started.\n")
    
    print("What would you like to do?\n")
    print("1. 🌐 Start Web Interface (Recommended)")
    print("2. 🧪 Run Test Suite")
    print("3. 📚 Run Example Usage")
    print("4. 📦 Install Dependencies")
    print("5. ℹ️  Show Information")
    print("6. ❌ Exit")
    
    choice = input("\nEnter your choice (1-6): ").strip()
    
    if choice == '1':
        start_web_interface()
    elif choice == '2':
        run_tests()
    elif choice == '3':
        run_examples()
    elif choice == '4':
        install_dependencies()
    elif choice == '5':
        show_info()
    elif choice == '6':
        print("\n👋 Goodbye!\n")
        sys.exit(0)
    else:
        print("\n❌ Invalid choice. Please try again.\n")
        main()

def start_web_interface():
    """Start the Flask web application"""
    print("\n" + "=" * 70)
    print("🌐 STARTING WEB INTERFACE")
    print("=" * 70)
    
    print("\nThe web interface will start on: http://localhost:5000")
    print("\nFeatures:")
    print("  • Upload gene expression datasets")
    print("  • Validate data quality")
    print("  • Configure preprocessing options")
    print("  • Download processed data")
    print("  • Collect data from GEO database")
    
    print("\nPress Ctrl+C to stop the server when done.\n")
    
    input("Press Enter to continue...")
    
    try:
        import app
        app.app.run(host='0.0.0.0', port=5000, debug=True)
    except KeyboardInterrupt:
        print("\n\n✅ Server stopped.\n")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure dependencies are installed:")
        print("  pip install -r requirements.txt\n")

def run_tests():
    """Run the test suite"""
    print("\n" + "=" * 70)
    print("🧪 RUNNING TEST SUITE")
    print("=" * 70)
    print()
    
    try:
        os.system('python test_system.py')
    except Exception as e:
        print(f"❌ Error running tests: {e}\n")

def run_examples():
    """Run example usage"""
    print("\n" + "=" * 70)
    print("📚 RUNNING EXAMPLES")
    print("=" * 70)
    print()
    
    try:
        os.system('python example_usage.py')
    except Exception as e:
        print(f"❌ Error running examples: {e}\n")

def install_dependencies():
    """Install required packages"""
    print("\n" + "=" * 70)
    print("📦 INSTALLING DEPENDENCIES")
    print("=" * 70)
    print()
    
    print("This will install:")
    print("  • pandas, numpy, scipy")
    print("  • scikit-learn")
    print("  • Flask, Flask-CORS")
    print("  • openpyxl (for Excel files)")
    print()
    
    confirm = input("Continue? (y/n): ").lower()
    if confirm == 'y':
        try:
            os.system('pip install -r requirements.txt')
            print("\n✅ Dependencies installed successfully!\n")
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
    else:
        print("\n❌ Installation cancelled.\n")

def show_info():
    """Show system information"""
    print("\n" + "=" * 70)
    print("ℹ️  SYSTEM INFORMATION")
    print("=" * 70)
    
    print("\n📁 Project Structure:")
    print("""
    gene/
    ├── app.py                  # Web application (Flask)
    ├── config.py              # Configuration settings
    ├── data_collector.py      # Data collection module
    ├── preprocessor.py        # Preprocessing pipeline
    ├── example_usage.py       # Usage examples
    ├── test_system.py         # Test suite
    ├── quick_start.py         # This script
    ├── requirements.txt       # Dependencies
    ├── README.md             # Documentation
    │
    ├── templates/
    │   └── index.html        # Web interface
    │
    ├── data/
    │   ├── raw/              # Raw datasets
    │   └── processed/        # Preprocessed data
    │
    ├── uploads/              # Uploaded files
    └── models/              # ML models (future)
    """)
    
    print("\n🎯 Supported Features:")
    print("  ✅ Multi-format upload (CSV, Excel, TXT)")
    print("  ✅ Data validation and preview")
    print("  ✅ Missing value handling (mean, median, KNN)")
    print("  ✅ Outlier detection (IQR, Z-score, Isolation Forest)")
    print("  ✅ Normalization (Z-score, Min-Max, Robust)")
    print("  ✅ Automatic data splitting (train/val/test)")
    print("  ✅ Label encoding")
    print("  ✅ GEO data collection")
    print("  ✅ Real-time preprocessing")
    
    print("\n🦠 Supported Diseases:")
    print("  • Breast Cancer")
    print("  • Lung Cancer")
    print("  • Prostate Cancer")
    print("  • Alzheimer's Disease")
    print("  • Parkinson's Disease")
    
    print("\n📚 Quick Commands:")
    print("  • Start web UI:     python app.py")
    print("  • Run tests:        python test_system.py")
    print("  • See examples:     python example_usage.py")
    print("  • Install deps:     pip install -r requirements.txt")
    
    print("\n🌐 Web Interface:")
    print("  URL: http://localhost:5000")
    print("  Features: Upload, Validate, Preprocess, Download")
    
    print("\n📖 Documentation:")
    print("  See README.md for detailed usage instructions")
    
    input("\n\nPress Enter to return to main menu...")
    print()
    main()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!\n")
        sys.exit(0)
