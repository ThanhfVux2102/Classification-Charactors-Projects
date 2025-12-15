# AG News – DistilBERT Project Interview Guide
## (HR • HR Tech • Technical Interview – Full Preparation)

---

## 1. One-line Abstract (Quick Intro)

> End-to-end NLP text classification project using the AG News dataset, covering EDA, preprocessing, Transformer fine-tuning (DistilBERT), evaluation, and deployment-ready model export.

---

## 2. Project Overview (2–3 minutes version)

This project focuses on building an end-to-end news classification system.  
The goal is to automatically classify news articles into four categories: World, Sports, Business, and Sci/Tech.

I handled the full pipeline, including exploratory data analysis, preprocessing, model training using a Transformer-based approach, evaluation with appropriate metrics, and exporting the trained model for backend deployment.

---

## 3. Dataset & EDA (Data Science Focus)

**Dataset**
- Source: Hugging Face Datasets – AG News
- Size: ~120k training samples, ~7.6k test samples
- Format: Arrow/Parquet (cached locally)

**EDA Performed**
- Class distribution analysis
- Text length distribution
- Vocabulary frequency inspection
- Data quality checks

EDA helped determine the max sequence length (128 tokens) and confirmed that accuracy and macro F1-score are suitable metrics.

---

## 4. Preprocessing & Tokenization

- Uses pretrained DistilBERT tokenizer
- Converts raw text into numerical representations (input_ids, attention_mask)
- Tokenization does not create tensors

Tensor conversion happens later, depending on the training framework.

Preprocessing is separated into a dedicated notebook to improve reproducibility and scalability.

---

## 5. Model & Training Strategy

- Model: DistilBERT
- Framework: TensorFlow + Hugging Face integration
- Data pipeline: prepare_tf_dataset for correct input formatting

This approach avoids manual tensor slicing errors and is production-friendly.

---

## 6. Evaluation & Metrics

- Accuracy
- Macro F1-score (treats all classes equally)

---

## 7. Model Export & Deployment Readiness

- Model and tokenizer exported using save_pretrained
- Compatible with FastAPI and Streamlit deployment

Inference pipeline:
User text → tokenizer → model → predicted label

---

## 8. Interview Strategy by Role

### HR Interview
Focus on:
- Business problem
- End-to-end ownership
- Real-world applicability

### HR Tech Interview
Focus on:
- Workflow
- Design decisions
- Deployment readiness

### Technical Interview
Focus on:
- Tokenization vs tensorization
- Training pipeline
- Metric selection
- Model export format

---

## 9. Common Follow-up Questions

Why DistilBERT?  
Good balance between performance and efficiency.

Can it run in real-time?  
Yes, suitable for real-time inference.

What would you improve?  
Hyperparameter tuning, model comparison, monitoring.

---

## 10. Closing Statement

This project demonstrates my ability to handle the full machine learning lifecycle, from data understanding to deployment preparation.

---

## 11. Interview Mindset

HR: Product thinking  
HR Tech: System thinking  
Technical: Engineering thinking
