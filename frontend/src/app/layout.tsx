import type { Metadata } from "next";
import { Jost } from "next/font/google";
import "./globals.css";

const fontSans = Jost({
  subsets: ["latin"],
  variable: "--font-sans",
});

export const metadata: Metadata = {
  title: "VisionNet Инференс",
  description: "Клиент API семантической сегментации",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ru">
      <body
        className={`${fontSans.variable} font-sans antialiased bg-background dark`}
      >
        {children}
      </body>
    </html>
  );
}
