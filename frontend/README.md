# SmartLock Frontend

Next.js UI for the local OpenCV smart-lock backend.

## Run

```bash
cd /home/tamminh/SmartLock_Fuzzy/frontend
npm install
cp .env.example .env.local
npm run dev
```

Open `http://localhost:3000`.

The app polls `NEXT_PUBLIC_API_URL/api/v1/camera/frame` for the live frame, detections, registration status, and fuzzy lock decision.
