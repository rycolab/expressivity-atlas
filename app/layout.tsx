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
        <div className="relative h-screen">
          {/* Sidebar is fixed, not in a flex row */}
          <SideBar />
          {/* Main content */}
          <div
            className="
              h-full
              transition-all duration-300
              md:pl-0
              "
            // If you want main content to "move" when sidebar is visible, 
            // you can use md:pl-64 and coordinate sidebar state with layout
          >
            <div className="p-6 md:overflow-y-auto md:p-12 h-full">
              <MathJaxContext config={mathJaxConfig}>{children}</MathJaxContext>
            </div>
          </div>
        </div>
      </body>
    </html>
  );
}
