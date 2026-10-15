import type { Metadata } from "next";
import "./globals.css";
import { Nav } from "@/components/Nav";

export const metadata: Metadata = {
  title: "Adaptive Graph + Vector RAG",
  description: "Inspectable adaptive retrieval: router, evidence, graph paths, citations, verification, costs.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header className="topbar">
          <div className="brand">
            Adaptive Graph + Vector RAG <span>· retrieval you can inspect</span>
          </div>
          <Nav />
        </header>
        <main className="page">{children}</main>
      </body>
    </html>
  );
}
