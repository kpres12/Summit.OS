import type { Metadata } from "next";
import { IBM_Plex_Mono, Orbitron } from "next/font/google";
import "./globals.css";
import { AuthProvider } from "../components/AuthProvider";
import PolicyNotifications from "../components/notifications/PolicyNotifications";

const ibmPlexMono = IBM_Plex_Mono({
  variable: "--font-ibm-plex-mono",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600"],
});

const orbitron = Orbitron({
  variable: "--font-orbitron",
  subsets: ["latin"],
  weight: ["400", "500", "700", "900"],
});

export const metadata: Metadata = {
  title: "Summit.OS",
  description: "Autonomous systems coordination platform for real-world operations",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${ibmPlexMono.variable} ${orbitron.variable} antialiased`}
      >
        <AuthProvider>
          {children}
          {/* Global policy notifications */}
          <PolicyNotifications />
        </AuthProvider>
      </body>
    </html>
  );
}
