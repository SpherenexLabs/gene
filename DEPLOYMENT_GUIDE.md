# Disease Gene Detection System - Deployment Guide

## 🚀 Deploy Your Project and Get a Public Link

Your project is now ready for deployment! Follow these simple steps:

---

## Option 1: Deploy to Render (FREE & RECOMMENDED)

### Step 1: Create a GitHub Repository
1. Go to [GitHub](https://github.com) and create a new repository
2. Name it something like `gene-detection-system`
3. **Don't** initialize with README (we already have files)

### Step 2: Push Your Code to GitHub
Open PowerShell in your project folder and run:

```powershell
# Initialize git repository
git init

# Add all files
git add .

# Commit your changes
git commit -m "Initial commit - Disease Gene Detection System"

# Add your GitHub repository (replace with your actual repo URL)
git remote add origin https://github.com/YOUR-USERNAME/gene-detection-system.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 3: Deploy on Render
1. Go to [Render.com](https://render.com) and sign up/login (use GitHub to sign in)
2. Click **"New"** → **"Web Service"**
3. Connect your GitHub repository
4. Render will auto-detect the settings from `render.yaml`
5. Click **"Create Web Service"**

### Step 4: Get Your Public Link! 🎉
- Render will build and deploy your app (takes 3-5 minutes)
- You'll get a URL like: `https://gene-detection-system.onrender.com`
- **This link works forever** - share it with anyone!

---

## Option 2: Deploy to Railway (Also FREE)

### Quick Deploy:
1. Go to [Railway.app](https://railway.app)
2. Sign up with GitHub
3. Click **"New Project"** → **"Deploy from GitHub repo"**
4. Select your repository
5. Add environment variable: `PORT` = `5000`
6. Railway will deploy automatically
7. Get your link: `https://your-app.up.railway.app`

---

## Option 3: Quick Test with ngrok (Instant Link)

For immediate testing without GitHub:

```powershell
# Install ngrok (if not installed)
# Download from: https://ngrok.com/download

# Start your Flask app
python app.py

# In another terminal, create public tunnel
ngrok http 5000
```

You'll get an instant link like: `https://abc123.ngrok-free.app`

⚠️ Note: ngrok links are temporary (expire when you close the terminal)

---

## Option 4: Deploy to PythonAnywhere (FREE)

1. Go to [PythonAnywhere.com](https://www.pythonanywhere.com) and sign up
2. Upload your project files
3. Set up a web app with Flask
4. Get your link: `https://yourusername.pythonanywhere.com`

---

## 📋 Files Created for Deployment:

✅ `Procfile` - Tells the server how to run your app
✅ `runtime.txt` - Specifies Python version
✅ `render.yaml` - Render.com configuration
✅ `.gitignore` - Excludes unnecessary files from git
✅ Updated `requirements.txt` - Added gunicorn web server
✅ Updated `config.py` - Supports dynamic PORT configuration

---

## 🔧 Troubleshooting:

**App won't start?**
- Check the deployment logs on Render/Railway dashboard
- Ensure all dependencies in `requirements.txt` are correct
- Verify Python version compatibility

**Link not working?**
- Wait 3-5 minutes for initial deployment
- Check if the service is "Running" in your dashboard
- Try accessing `/api/health` endpoint first

**Upload not working?**
- Free tier platforms may have storage limitations
- Consider using external storage (AWS S3, Cloudinary) for large files

---

## 🎯 Recommended: Use Render

**Why Render?**
- ✅ Completely FREE tier
- ✅ Automatic HTTPS
- ✅ Easy GitHub integration
- ✅ No credit card required
- ✅ Link never expires
- ✅ Auto-redeploys on git push

Your app will be live at: `https://[your-app-name].onrender.com`

---

## Need Help?

If you run into issues:
1. Check deployment logs in your hosting dashboard
2. Verify all files are committed to GitHub
3. Ensure Python version matches `runtime.txt`

Good luck with your deployment! 🚀
