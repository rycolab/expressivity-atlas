import '@/app/ui/global.css';
import { inter } from '@/app/ui/fonts';
import SideBar from '@/app/ui/sidebar';
import { MathJaxContext } from 'better-react-mathjax';
import mathJaxConfig from '@/mathjax-config';

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={`${inter.className} antialiased`}>
        <div className="flex h-screen flex-col md:flex-row md:overflow-hidden">
            <div className="w-full flex-none md:w-64">
                <SideBar />
            </div>
            <div className="flex-grow p-6 md:overflow-y-auto md:p-12">
              <MathJaxContext config={mathJaxConfig}>{children}</MathJaxContext>
            </div>
        </div>
      </body>
    </html>
  );
}
