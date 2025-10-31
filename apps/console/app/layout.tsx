import type { Metadata } from "next";
import { IBM_Plex_Mono } from "next/font/google";
import "./globals.css";
import { AuthProvider } from "../components/AuthProvider";
import dynamic from "next/dynamic";

const ibmPlexMono = IBM_Plex_Mono({
  variable: "--font-ibm-plex-mono",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600"],
});

export const metadata: Metadata = {
  title: "Summit.OS Command Interface",
  description: "Tactical operations control center for autonomous systems coordination",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${ibmPlexMono.variable} antialiased`}
      >
        <AuthProvider>
          {children}
          {/* Global policy notifications */}
          {dynamic(() => import('../components/notifications/PolicyNotifications'), { ssr: false })({})}
        </AuthProvider>
      </body>
    </html>
  );
}
