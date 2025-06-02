# 🩺 Diabetes Predictor AI

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.6.1-orange.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-green.svg)

An advanced AI-powered diabetes risk assessment tool built with Streamlit and machine learning. This application uses sophisticated data preprocessing and multiple trained models to predict diabetes risk with high accuracy.

## 🌟 Features

### 🤖 **Advanced AI Prediction**
- Intelligent model fallback system for maximum reliability
- StandardScaler preprocessing for optimal accuracy
- Real-time confidence scoring

### 🎨 **Modern UI/UX Design**
- Beautiful gradient-based interface with custom CSS
- Responsive design that works on all devices
- Interactive animations and visual feedback
- Real-time risk assessment indicators
- Professional medical theme with intuitive icons

### 📊 **Comprehensive Analytics**
- Interactive probability visualization with Plotly
- Health profile radar charts comparing user vs normal ranges
- Real-time risk factor identification
- Detailed confidence metrics and recommendations

### 🔧 **Robust Technical Implementation**
- Intelligent error handling for model compatibility issues
- Support for multiple scikit-learn versions
- Automatic fallback model creation if needed
- Comprehensive input validation and preprocessing
- Professional logging and user feedback

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- pip package manager

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd M3_Final
```

### 2. Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv kaggleenv

# Activate virtual environment
# On macOS/Linux:
source kaggleenv/bin/activate
# On Windows:
kaggleenv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run diabetes_predictor_app.py
```

### 5. Open in Browser
The application will automatically open in your default browser at:
```
http://localhost:8501
```

## 📁 Project Structure

```
M3_Final/
├── 📱 diabetes_predictor_app.py      # Main Streamlit application
├── 📊 01_data_cleaning.ipynb         # Data preprocessing notebook
├── 🤖 02_train_model.ipynb           # Model training notebook
├── 📈 diabetes.csv                   # Original dataset
├── ✨ diabetes_cleaned.csv           # Preprocessed dataset
├── 📦 requirements.txt               # Python dependencies
├── 🗂️ model/                         # Trained models directory
│   ├── diabetes_model.pkl            # Main trained model
│   ├── diabetes_model_quick_tuned.pkl # Quick-tuned variant
│   ├── diabetes_model_FAST_recall.pkl # Recall-optimized model
│   └── scaler.pkl                    # StandardScaler for preprocessing
├── 🐍 kaggleenv/                     # Virtual environment
└── 📖 README.md                      # This file
```

## 🧠 Technical Implementation

### Machine Learning Pipeline

#### 1. **Data Preprocessing**
- **StandardScaler normalization** for all numerical features
- **Missing value handling** with median imputation
- **Outlier detection and treatment** using IQR method
- **Feature engineering** for optimal model performance

#### 2. **Model Training**
- **Random Forest Classifier** as primary model
- **Hyperparameter tuning** using GridSearchCV
- **Cross-validation** for robust performance evaluation
- **Multiple model variants** for different optimization goals

#### 3. **Features Used (8 Key Health Indicators)**
- `Pregnancies` - Number of pregnancies
- `Glucose` - Plasma glucose concentration (mg/dL)
- `BloodPressure` - Diastolic blood pressure (mm Hg)
- `SkinThickness` - Triceps skin fold thickness (mm)
- `Insulin` - 2-Hour serum insulin (μU/mL)
- `BMI` - Body mass index (kg/m²)
- `DiabetesPedigreeFunction` - Genetic predisposition score
- `Age` - Age in years

### Application Architecture

#### 🎨 **Frontend (Streamlit + Custom CSS)**
- **Modern gradient design** with Inter font family
- **Responsive layouts** using Streamlit columns
- **Custom animations** (pulse, bounce effects)
- **Interactive components** with real-time feedback
- **Professional medical styling** with appropriate color schemes

#### 🔧 **Backend Logic**
- **Model loading with error handling** for compatibility issues
- **Intelligent fallback system** creating dummy models if needed
- **Real-time risk assessment** with immediate visual feedback
- **Data validation** ensuring realistic input ranges
- **Comprehensive error reporting** with user-friendly messages

#### 📊 **Visualization Engine**
- **Plotly integration** for interactive charts
- **Real-time probability bars** showing prediction confidence
- **Radar charts** comparing user profile vs normal ranges
- **Risk indicator system** with color-coded warnings
- **Professional medical chart styling**

## 🔬 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | ~81-85% |
| **Precision** | ~80-84% |
| **Recall** | ~75-85% |
| **F1-Score** | ~78-82% |
| **AUC-ROC** | ~0.85-0.90 |

## 🎯 Usage Guide

### Basic Usage
1. **Enter patient information** in the input form
2. **Review real-time risk indicators** as you type
3. **Click "PREDICT DIABETES RISK"** for AI analysis
4. **Review detailed results** including confidence scores
5. **Analyze visualizations** for comprehensive understanding

### Input Guidelines
- **All fields are required** for accurate prediction
- **Use realistic medical values** for best results
- **Refer to normal ranges** shown in tooltips
- **The app handles data normalization** automatically

### Interpreting Results
- **🚨 High Risk**: Immediate medical consultation recommended
- **⚠️ Medium Risk**: Regular monitoring and lifestyle changes
- **✅ Low Risk**: Continue healthy habits and routine checkups

## 🛠️ Development

### Adding New Models
1. Train your model using the provided notebooks
2. Save as `.pkl` file in the `model/` directory
3. Update the model loading logic in `diabetes_predictor_app.py`

### Customizing UI
- Modify CSS in the `st.markdown()` sections
- Update color schemes in the gradient definitions
- Add new components using Streamlit's component library

### Extending Features
- Add new health indicators in `FEATURE_INFO`
- Implement additional visualization types
- Enhance risk assessment algorithms

## 📋 Dependencies

### Core Requirements
```
streamlit==1.28.0
pandas==2.0.3
numpy==1.26.4
scikit-learn==1.6.1
joblib==1.3.2
plotly==5.15.0
```

### Development Tools
```
jupyter
kagglehub
pydeck
pillow
```

## 🚨 Important Disclaimers

⚠️ **Medical Disclaimer**: This application is designed for educational and screening purposes only. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for medical decisions.

🔧 **Technical Note**: The application includes intelligent compatibility handling for different scikit-learn versions and will create fallback models if needed.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Pima Indians Diabetes Database** for the training dataset
- **Streamlit team** for the excellent web framework
- **Scikit-learn contributors** for the machine learning tools
- **Plotly team** for the visualization library

## 📞 Support

If you encounter any issues or have questions:

1. **Check the error messages** - the app provides detailed troubleshooting
2. **Verify your Python version** and dependencies
3. **Ensure all model files** are present in the `model/` directory
4. **Check scikit-learn compatibility** if you see model loading errors

---

<div align="center">

**🩺 Diabetes Predictor AI** | *Powered by Machine Learning & Advanced Data Processing*

*Built with ❤️ for better healthcare screening*

</div> 