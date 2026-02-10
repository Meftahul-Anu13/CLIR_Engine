import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CLIR Search",
  description: "Cross-lingual information retrieval demo",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
