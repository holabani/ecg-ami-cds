import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import AuthGate from '@/components/AuthGate';
import './globals.css';

const inter = Inter({ subsets: ['latin'], variable: '--font-geist-sans' });

export const metadata: Metadata = {
  title: 'ECG AMI CDS | AI-Based Clinical Decision Support',
  description: 'Early Detection of AMI and Revascularization Need',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="min-h-screen bg-[#0f0f1a] text-gray-100 antialiased">
        <AuthGate>{children}</AuthGate>
      </body>
    </html>
  );
}
