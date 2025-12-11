# Quick Start Guide - NLP Final Project

## ⚡ 5-Minute Setup

### Step 1: Install Dependencies (2 minutes)

Open a terminal/command prompt and run:

```bash
pip install transformers datasets torch gradio bertviz wordcloud matplotlib seaborn pandas numpy accelerate evaluate scikit-learn peft
```

### Step 2: Open the Notebook (1 minute)

```bash
# Navigate to the project folder
cd "c:\Users\sruth\Documents\SEM 4\NLP\Final"

# Launch Jupyter
jupyter notebook NLP_Final_Project_QA_System.ipynb
```

### Step 3: Run the Notebook (2 minutes)

1. Click **Cell** → **Run All** in Jupyter
2. Wait for execution to complete
3. Interact with the Gradio demo at the end!

---

## 🎯 What Each Section Does

### Section 1: Data Preparation (5-10 min runtime)
- Downloads SQuAD 2.0 dataset automatically
- Shows sample questions and answers
- Creates visualizations (histograms, word clouds)
- Preprocesses data for training

**Output**: `data_exploration.png`

### Section 2: Extractive QA (15-30 min runtime)
- Fine-tunes DistilBERT on question answering
- Trains for 3 epochs
- Evaluates with Exact Match and F1 scores
- Shows sample predictions

**Output**: `qa_model_final/`, `qa_training_curves.png`

### Section 3: Response Generation (10-20 min runtime)
- Fine-tunes DistilGPT-2 for explanations
- Creates prompt-response pairs
- Generates sample explanations
- Calculates perplexity

**Output**: `gpt2_generation_final/`, `gpt2_training_curves.png`

### Section 4: Advanced Exploration (10-15 min runtime)
- Visualizes attention mechanisms
- Compares zero-shot vs fine-tuned
- Demonstrates LoRA efficiency

**Output**: `attention_heatmap.png`

### Section 5: Interactive Demo (instant)
- Launches Gradio web interface
- Try your own questions!
- Deployable to Hugging Face Spaces

**Output**: Live web interface

### Section 6: Analysis (instant)
- Model comparison tables
- Performance visualizations
- Discussion and reflection

**Output**: `performance_comparison.png`, `model_comparison.csv`

---

## 💡 Pro Tips

### If You Have Limited Time
Run only these sections:
1. Section 1 (Data Preparation) - Required
2. Section 2 (Extractive QA) - Core functionality
3. Section 5 (Demo) - See it in action!

### If You Have Limited GPU/RAM
Look for this line in the notebook:
```python
USE_SUBSET = True  # Already set by default
```
This uses only 10,000 training samples instead of 87,000.

### If Training is Too Slow
Reduce epochs:
```python
num_train_epochs = 2  # Instead of 3
```

Reduce batch size:
```python
per_device_train_batch_size = 8  # Instead of 16
```

---

## 🔍 Expected Timeline

| Hardware | Total Runtime |
|----------|---------------|
| GPU (NVIDIA RTX 3060+) | 30-45 minutes |
| GPU (NVIDIA GTX 1060) | 60-90 minutes |
| CPU only | 3-5 hours |

**Recommendation**: Use Google Colab (free GPU) if you don't have a GPU:
1. Upload notebook to Google Drive
2. Open with Google Colab
3. Runtime → Change runtime type → GPU

---

## 📊 What Success Looks Like

### Extractive QA Metrics
```
Exact Match (EM): 70-80%
F1 Score: 75-85%
```

### Generation Metrics
```
Perplexity: 15-25
```

### Sample Output
```
Question: "Which country contains the majority of the Amazon rainforest?"
Predicted Answer: "Brazil"
Confidence: 98.5%
Explanation: "Based on the context, the answer to 'Which country contains 
the majority of the Amazon rainforest?' is 'Brazil'. This can be found in 
the passage where it states that Brazil contains 60% of the rainforest."
```

---

## 🆘 Common Issues & Solutions

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size to 8 or 4

### Issue: "Module not found"
**Solution**: Run `pip install [module_name]`

### Issue: "Dataset download fails"
**Solution**: Check internet connection, retry

### Issue: "Training is very slow"
**Solution**: 
- Set `USE_SUBSET = True`
- Reduce `num_train_epochs` to 2
- Use GPU or Google Colab

### Issue: "Gradio demo won't launch"
**Solution**: 
- Make sure all previous cells ran successfully
- Check that models are loaded
- Try `demo.launch(share=False, debug=True)`

---

## 🎓 Understanding the Output

### Files Created

1. **PNG Images**: Visualizations you can include in reports
2. **CSV File**: Metrics table for analysis
3. **Model Folders**: Saved models you can reload later

### Reloading Saved Models

```python
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# Load your fine-tuned model
model = AutoModelForQuestionAnswering.from_pretrained("./qa_model_final")
tokenizer = AutoTokenizer.from_pretrained("./qa_model_final")

# Use for predictions
from transformers import pipeline
qa = pipeline("question-answering", model=model, tokenizer=tokenizer)

result = qa(
    question="What is the capital of France?",
    context="Paris is the capital and largest city of France."
)
print(result)
```

---

## 🚀 Next Steps After Completion

### 1. Experiment
- Try different questions in the Gradio demo
- Test on your own passages
- Adjust hyperparameters and retrain

### 2. Deploy
- Upload to Hugging Face Spaces
- Share with friends/classmates
- Add to your portfolio

### 3. Extend
- Fine-tune on domain-specific data (medical, legal, etc.)
- Add multilingual support
- Implement retrieval-augmented generation (RAG)

### 4. Document
- Export notebook as PDF for submission
- Take screenshots of the Gradio demo
- Save all generated visualizations

---

## 📝 Submission Checklist

Before submitting, ensure you have:

- [ ] Jupyter notebook (.ipynb file)
- [ ] PDF export of the notebook
- [ ] All generated visualizations (PNG files)
- [ ] Model comparison CSV
- [ ] Link to deployed Gradio demo (if deployed)
- [ ] README.md (this file)

---

## 🎉 You're Ready!

Just run the notebook and watch the magic happen. The code is well-commented and designed to run smoothly from start to finish.

**Good luck with your final project! 🚀**

---

## 📞 Need Help?

If you encounter issues:
1. Check the error message carefully
2. Review the "Common Issues" section above
3. Make sure all dependencies are installed
4. Try restarting the Jupyter kernel
5. Check that you have enough disk space (~2GB needed)

**Remember**: The notebook is designed to be self-contained and educational. Read the markdown cells for explanations of what each code block does!

