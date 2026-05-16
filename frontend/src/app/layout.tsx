import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'SmartLock Face Recognition',
  description: 'Local OpenCV camera smart-lock interface',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="font-sans">{children}</body>
    </html>
  )
}
