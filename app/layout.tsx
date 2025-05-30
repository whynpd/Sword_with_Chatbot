import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { DiscordContextProvider } from '@/contexts/DiscordContext';
import { ClerkProvider } from '@clerk/nextjs';
import ChatBot from '@/components/ChatBot/ChatBot';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Discord Clone',
  description: 'Powered by Stream Chat',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <ClerkProvider>
      <html lang='en'>
        <DiscordContextProvider>
          <body className={inter.className}>
            {children}
            <ChatBot />
          </body>
        </DiscordContextProvider>
      </html>
    </ClerkProvider>
  );
}
