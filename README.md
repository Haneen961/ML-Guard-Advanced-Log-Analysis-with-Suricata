# ML-Guard-Advanced-Log-Analysis-with-Suricata

ML-Guard: Advanced Log Analysis with Suricata is a comprehensive security analysis framework designed to detect and monitor cyber threats using AI-driven detection models and Suricata-based intrusion analysis.

📌 Project Structure
Our project consists of several key components:

AI Algorithms & Models

Machine learning models trained for detecting SQL Injection (SQLi), Cross-Site Scripting (XSS), and Layer 7 DDoS attacks.
Supports various attack detection mechanisms with high accuracy.
Can be adapted to detect different variations of these attacks by modifying datasets and feature selection.
Suricata Rules & Configuration

Custom Suricata rules for detecting SQLi, XSS, and Layer 7 DDoS threats.
Configurable to enhance detection precision.
Backend Python Scripts

Handles log monitoring, feature extraction, and preprocessing.
Connects with AI models for real-time threat evaluation.
Communicates with the front-end app & web dashboard for live analysis and visualization.
🚀 Adapting the Project for Different Attack Variants
If you want to extend or refine the detection of SQLi, XSS, or Layer 7 DDoS attacks, you can:
✅ Use our AI algorithms but modify the dataset and feature engineering process.
✅ Train models with new patterns of SQLi, XSS, or DDoS attacks.

⚡ Using Pre-Trained Models
If you want to use our pre-trained models:
1️⃣ Download the trained models from the repository.
2️⃣ Use the backend scripts to monitor logs and pass them to the AI models.
3️⃣ The AI models will evaluate the logs and identify potential SQLi, XSS, or Layer 7 DDoS threats.

This approach ensures easy integration and adaptability for different cybersecurity scenarios.
