import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "TherA Lite",
  description: "Client-side AI thermal reasoning prototype for RGB images."
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
