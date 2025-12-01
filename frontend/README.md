# Non-Invasive Glucose Monitor - Frontend

⚠️ **MEDICAL DISCLAIMER**: This is a research demonstration ONLY. NOT a medical device. NOT for clinical use.

## Setup

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Configure API URL:**
   Create a `.env.local` file:
   ```
   NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
   ```

3. **Run development server:**
   ```bash
   npm run dev
   ```

4. **Open in browser:**
   Navigate to `http://localhost:3000`

## Deployment to Vercel

1. **Install Vercel CLI:**
   ```bash
   npm install -g vercel
   ```

2. **Deploy:**
   ```bash
   vercel
   ```

3. **Set environment variable in Vercel:**
   - Go to your project settings on Vercel
   - Add environment variable:
     - Name: `NEXT_PUBLIC_API_BASE_URL`
     - Value: Your deployed backend URL (e.g., `https://your-api.railway.app`)

4. **Redeploy:**
   ```bash
   vercel --prod
   ```

## Project Structure

```
frontend/
├── app/
│   ├── layout.tsx          # Root layout with metadata
│   ├── page.tsx            # Main prediction page
│   └── globals.css         # Global styles
├── components/
│   ├── DisclaimerBanner.tsx    # Medical disclaimer banner
│   ├── PredictionForm.tsx      # Input form (demo/manual)
│   └── ResultsDisplay.tsx      # Results with color-coded ranges
├── lib/
│   └── api.ts              # API client functions
└── package.json
```

## Features

- **Disclaimer Banner**: Prominent medical disclaimer at top
- **Demo Mode**: Select from 100 pre-loaded test samples
- **Manual Mode**: Enter basic features (HR, HRV, time since meal)
- **Results Display**: Color-coded glucose ranges (low/normal/high)
- **Responsive Design**: Works on desktop and mobile
- **Type-Safe**: Full TypeScript support

## Important Notes

- This is a **research demonstration** using synthetic data
- **NOT** for medical use or clinical decisions
- **NOT** FDA approved or clinically validated
- Always consult healthcare professionals for medical advice
