import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'A little something, just for you',
  description: 'A little birthday surprise, made with a lot of thought.',
  robots: { index: false, follow: false },
};
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
