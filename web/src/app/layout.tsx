import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Custodia — agent memory with a chain of custody",
  description:
    "Every fact bound to the turn it came from, every answer cited, and a refusal when memory has nothing to stand on.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
