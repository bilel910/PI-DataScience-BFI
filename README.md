# Automated Financial Statement Processing (ESPRIT/BFI)

This project leverages advanced deep-learning methodologies, including YOLOv8 architecture, combined with modern visualization tools, to automate financial statement processing and enhance user accessibility. By addressing challenges in document complexity, data extraction, and decision-making, this system provides a cutting-edge solution to streamline workflows in financial data management.

## Repository  
[GitHub Repository](https://github.com/bilel910/PI-DataScience-BFI)

## Project Structure

### 1. Business Understanding
- **Introduction**: Overview of the manual processes and challenges in financial data processing.
- **Study of the Existing System**: Detailed examination of issues, including:
  - Manual data entry.
  - Complexity of financial documents.
  - Lack of standardization.
  - Time-consuming analysis.
  - Dependence on human interpretation.
  - High operational costs and security concerns.
- **Business Problem**: Highlighting inefficiencies due to manual processes and document complexity.
- **Business Objectives**:
  - Improve efficiency and reduce operational costs.
  - Automate data entry and ensure data consistency.
  - Enhance decision-making capabilities.
- **Data Science Objectives**:
  - Use AI models to improve accuracy and processing speed.
  - Ensure standardization in data handling.

---

### 2. Data Acquisition and Understanding
- **Data Sources**: Collected from CMF, BVMT, and BRVM repositories.
- **Data Preparation**:
  - Data extraction and filtering.
  - Table detection using YOLOv8 and DeepDoctection.
  - Data conversion for model compatibility.
- **Challenges**: Addressing data inconsistency, noise, and document complexity.

---

### 3. Modeling
- **Model Selection**:
  - Document Classification: YOLOv8, ResNet50.
  - Data Automation: YOLOv8 + Doctr, DeepDoctection, Tabula.
  - Decision-Making Models: ARIMA, SARIMA, Linear Regression.
- **Model Evaluation**:
  - Performance metrics including accuracy, precision, error rate, and processing speed.
- **Chatbot Integration**:
  - Using Langchain for document submission, prompt generation, and result visualization.

---

### 4. Deployment
- **Technology Stack**:
  - Flask for backend API development.
  - SQLite3 for database management.
  - Jinja2 for templating.
- **Deployment Modules**:
  - Authentication processes for secure access.
  - Automated data extraction and classification.
  - Visualization of statistics and charts.
  - Deployment of chatbot for interaction.
- **Monitoring and Maintenance**:
  - Regular updates for continuous improvement and system reliability.

---

### 5. Results and Outcomes
- Automated document processing with improved efficiency.
- Reduced error rates in financial data handling.
- Enhanced decision-making support with AI-powered analytics.

---

## Key Features
- **Automation**: Streamlines financial document processing.
- **Deep Learning**: YOLOv8 and other architectures for document classification.
- **Data Extraction**: Advanced tools like Doctr and DeepDoctection.
- **Chatbot Support**: Easy user interaction and real-time assistance.
- **Visualization**: Intuitive dashboards and charts for better insights.

---

## Technologies Used
- **Deep Learning Models**: YOLOv8, ResNet50, ARIMA.
- **Programming Languages**: Python.
- **Frameworks**: Flask, SQLite3, Langchain.
- **Visualization Tools**: Matplotlib, Seaborn.

---

## Challenges Addressed
- Document complexity and lack of standardization.
- Manual data entry and time-consuming analysis.
- Enhancing data security and adaptability.

---

## Future Perspectives
- Expand the system to include multilingual document support.
- Integrate advanced NLP models for enhanced document understanding.
- Deploy on scalable cloud infrastructure for larger datasets.

---

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/bilel910/PI-DataScience-BFI.git
   
2.Install dependencies:
  pip install -r requirements.txt
  
3.Run the application
  flask run
