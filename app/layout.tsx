import type { Metadata } from "next";
import "./globals.css";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";

export const metadata: Metadata = {
  title: "ChestPrior — Generative Priors for Robust Chest X-ray Classification",
  description: "Comparative Analysis of Generative Priors and Dual Feature Aggregation for Robust Chest X-ray Classification under Data Imbalance",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Navbar />
        <main style={{ position: "relative", zIndex: 1 }}>
          {children}
        </main>
        <Footer />
      </body>
    </html>
  );
}
