# GitHub Codespaces Troubleshooting Guide

## 🐛 Issue: Model Download Stuck at 0%

### Why This Happens in Codespaces
- Network bandwidth limitations
- Codespaces may throttle large downloads
- Progress bar doesn't always update in real-time
- Background downloads may not show progress

---

## ✅ Solution 1: Wait It Out (Recommended First)

**The download IS happening, even if stuck at 0%!**

1. **Wait 2-5 minutes** without interrupting
2. The progress bar may jump from 0% to 100% suddenly
3. Total download: ~268MB for DistilBERT

**How to verify it's downloading:**
```bash
# Open a new terminal and run:
watch -n 1 du -sh ~/.cache/huggingface/hub/
```
You should see the size increasing.

---

## ✅ Solution 2: Interrupt and Resume

If truly stuck after 5 minutes:

1. **Interrupt**: Press `Ctrl + C` in the notebook
2. **Re-run the cell**: The download will resume from where it stopped
3. Repeat if necessary

The code already has `resume_download=True` enabled.

---

## ✅ Solution 3: Use Pre-Fine-Tuned Model (Fastest)

In the notebook, **use OPTION 2** instead of OPTION 1:

1. **Comment out** OPTION 1 cell (add `#` at start of each line)
2. **Uncomment** OPTION 2 cell (remove `#` from each line)
3. Run OPTION 2

This uses `distilbert-base-uncased-distilled-squad` which:
- ✅ Smaller download
- ✅ Already fine-tuned on SQuAD
- ✅ Faster to load
- ✅ Still gets you great results

---

## ✅ Solution 4: Manual Download via Terminal

If notebook download fails, try terminal:

```bash
# In Codespaces terminal:
python3 << EOF
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

print("Downloading model...")
model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

print("✓ Download complete!")
print("Now run the notebook cell again - it will use cached version")
EOF
```

---

## ✅ Solution 5: Upgrade Hugging Face Hub

Sometimes the download library needs updating:

```bash
pip install --upgrade huggingface-hub transformers
```

Then restart the kernel and try again.

---

## ✅ Solution 6: Check Codespaces Resources

### Check available disk space:
```bash
df -h
```

You need at least **2GB free** for models and data.

### Check network connectivity:
```bash
curl -I https://huggingface.co
```

Should return `HTTP/2 200`.

---

## 🚀 Quick Fix Commands

Run these in a Codespaces terminal:

```bash
# 1. Upgrade packages
pip install --upgrade huggingface-hub transformers

# 2. Clear cache if needed (only if desperate)
rm -rf ~/.cache/huggingface/hub/

# 3. Pre-download model
python3 -c "from transformers import AutoModelForQuestionAnswering; AutoModelForQuestionAnswering.from_pretrained('distilbert-base-uncased')"

# 4. Verify download
ls -lh ~/.cache/huggingface/hub/
```

---

## 📊 Expected Download Times in Codespaces

| Model | Size | Expected Time |
|-------|------|---------------|
| distilbert-base-uncased | 268MB | 2-5 minutes |
| distilbert-base-uncased-distilled-squad | 260MB | 2-5 minutes |
| distilgpt2 | 353MB | 3-7 minutes |

**Note**: First download is slow, subsequent runs use cache (instant).

---

## 🔍 How to Monitor Download Progress

### Option 1: Watch cache directory
```bash
# In a new terminal:
watch -n 2 'du -sh ~/.cache/huggingface/hub/ && ls -lh ~/.cache/huggingface/hub/models--distilbert-base-uncased/blobs/ 2>/dev/null | tail -5'
```

### Option 2: Check network activity
```bash
# Monitor network usage:
iftop
# or
nethogs
```

### Option 3: Python progress
The notebook code now shows:
- Start time
- Elapsed time when complete
- Clear success/error messages

---

## 💡 Pro Tips for Codespaces

### 1. Use Local Cache
The updated code uses `cache_dir="./model_cache"` to keep models in your workspace.

### 2. Commit Cache to Repo (Optional)
```bash
# After successful download:
git add model_cache/
git commit -m "Add cached models"
git push
```
Next time you open Codespaces, models are already there!

### 3. Use Smaller Subset
In the data preparation section, keep:
```python
USE_SUBSET = True  # Uses 10k samples instead of 87k
```

### 4. Enable Verbose Logging
```python
import logging
logging.basicConfig(level=logging.INFO)
```

---

## 🆘 Still Stuck?

### Last Resort Options:

**Option A: Use Google Colab Instead**
1. Upload notebook to Google Drive
2. Open with Google Colab
3. Free GPU + faster downloads
4. Runtime → Change runtime type → GPU

**Option B: Run Locally**
1. Clone repo to your local machine
2. Install dependencies
3. Run notebook locally
4. Push results back to GitHub

**Option C: Use Smaller Models**
Replace in the notebook:
```python
model_checkpoint = "distilbert-base-uncased"
# Change to:
model_checkpoint = "prajjwal1/bert-tiny"  # Only 17MB!
```

---

## ✅ Verification Checklist

After model loads successfully:

- [ ] No error messages
- [ ] See "✅ SUCCESS! Model loaded"
- [ ] Parameter count displayed (~66M for DistilBERT)
- [ ] Device shows (cpu or cuda)
- [ ] Can proceed to next cells

---

## 📞 Getting Help

If none of these work:

1. **Check Codespaces status**: https://www.githubstatus.com/
2. **Restart Codespace**: Codespaces → Restart
3. **Create new Codespace**: Sometimes a fresh start helps
4. **Check quota**: Free tier has limited hours/month

---

## 🎯 Recommended Approach for Codespaces

**For fastest results:**

1. ✅ Use OPTION 2 (pre-fine-tuned model)
2. ✅ Keep `USE_SUBSET = True`
3. ✅ Run cells sequentially
4. ✅ Don't interrupt downloads
5. ✅ Monitor with terminal commands

**This will get you running in < 10 minutes!**

---

## 📝 Summary

| Issue | Solution | Time |
|-------|----------|------|
| Download stuck | Wait 2-5 min | 5 min |
| Still stuck | Ctrl+C, re-run | 2 min |
| Too slow | Use OPTION 2 | 3 min |
| Fails completely | Terminal download | 5 min |
| Desperate | Google Colab | 10 min |

**Most common fix**: Just wait! The download is happening. ⏳

Good luck! 🚀

