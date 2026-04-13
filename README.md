# ChestPrior Website

Website proyek KCV — Generative Priors & Dual Feature Aggregation for Robust Chest X-ray Classification.

## Deploy ke Vercel

### Via GitHub (Recommended)
1. Push ke GitHub repo
2. Buka vercel.com → New Project → Import repo
3. Deploy otomatis (settings Next.js terdeteksi sendiri)

### Via Vercel CLI
```bash
npm install -g vercel
vercel
```

## Local Dev
```bash
npm install
npm run dev
```

## Customization
- **Hasil eksperimen**: Edit `app/results/page.tsx` → ganti `TBD` dengan nilai aktual
- **Anggota tim**: Edit `app/team/page.tsx` → update array `team`
- **Comparison scores**: Edit `app/comparison/page.tsx` → update `mockScores` 
- **Nama project**: Ganti "ChestPrior" di `app/layout.tsx`

ITS Surabaya · KCV · 2026
