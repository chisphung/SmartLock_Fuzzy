# People Counting GUI

A Next.js web application for the People Counting system using YOLOv11.

## Features

- **Image Upload**: Drag-and-drop or click to upload images
- **Real-time Detection**: View bounding boxes around detected people
- **Statistics Dashboard**: Track detection metrics and history
- **Responsive Design**: Works on desktop and mobile devices

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn
- Backend API running on `http://localhost:8000`

### Installation

1. Install dependencies:
```bash
cd gui
npm install
```

2. Create environment file:
```bash
cp .env.example .env.local
```

3. Start the development server:
```bash
npm run dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Project Structure

```
gui/
├── src/
│   ├── app/
│   │   ├── globals.css      # Global styles
│   │   ├── layout.tsx       # Root layout
│   │   └── page.tsx         # Main page
│   ├── components/
│   │   ├── BoundingBoxCanvas.tsx  # Canvas for drawing detections
│   │   ├── Header.tsx             # App header
│   │   ├── ImageUploader.tsx      # Image upload component
│   │   └── StatsPanel.tsx         # Statistics panel
│   ├── lib/
│   │   └── api.ts           # API client
│   └── types/
│       └── index.ts         # TypeScript types
├── package.json
├── tailwind.config.ts
└── tsconfig.json
```

## API Endpoints Used

- `POST /api/v1/count/upload` - Upload and analyze image
- `POST /api/v1/count/base64` - Analyze base64 encoded image
- `GET /api/v1/result/{filename}` - Get result image

## Technologies

- **Next.js 14** - React framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **Axios** - HTTP client
