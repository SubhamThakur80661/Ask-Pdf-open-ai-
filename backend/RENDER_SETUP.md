# Render Manual Deployment Guide

This guide describes how to manually configure and deploy the OpCit backend as a Web Service on Render (free tier) if you are not using the `render.yaml` Blueprint.

## Step-by-Step Instructions

1. **Log in to Render**
   - Navigate to [dashboard.render.com](https://dashboard.render.com/) and sign in.

2. **Create a New Web Service**
   - Click **New +** in the top navigation bar and select **Web Service**.
   - Connect your GitHub repository containing the OpCit codebase.

3. **Configure Service Settings**
   - **Name**: `opcit-backend`
   - **Region**: Choose the region closest to your users (e.g., Oregon, Frankfurt).
   - **Branch**: `rebuild` (or `main` when merged).
   - **Root Directory**: `backend`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: `Free`

4. **Configure Health Check**
   - Expand the **Advanced** section.
   - Set **Health Check Path** to `/health`.

5. **Set Environment Variables**
   Under the **Environment Variables** section, add:
   - `PYTHON_VERSION`: `3.12.8`
   - `CORS_ORIGINS`: `http://localhost:5173,https://your-frontend.vercel.app` (replace with your actual frontend Vercel URL once deployed).
   - `HUGGINGFACEHUB_API_TOKEN`: *(Optional)* Add only if you wish to provide a global server-side Hugging Face fallback token. If left unset, users must supply their own token in the frontend Settings.

6. **Deploy**
   - Click **Create Web Service**.
   - Wait for the build and deployment to complete.
   - Once deployed, copy your backend service URL (e.g., `https://opcit-backend.onrender.com`).
   - Test the endpoint:
     ```bash
     curl https://opcit-backend.onrender.com/health
     # Expected response: {"status":"ok"}
     ```
