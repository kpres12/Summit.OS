'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '../../components/AuthProvider';

export default function LoginPage() {
  const { isAuthenticated, isLoading, login } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (isAuthenticated) {
      router.push('/');
    }
  }, [isAuthenticated, router]);

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#0A0A0A]">
        <div className="text-[#00FF91] text-xl font-mono animate-pulse">
          Loading...
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#0A0A0A] relative overflow-hidden">
      {/* Background grid effect */}
      <div className="absolute inset-0 opacity-10">
        <div
          className="w-full h-full"
          style={{
            backgroundImage:
              'linear-gradient(rgba(0,255,145,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(0,255,145,0.1) 1px, transparent 1px)',
            backgroundSize: '50px 50px',
          }}
        />
      </div>

      {/* Login Card */}
      <div className="relative z-10 w-full max-w-md p-8">
        <div
          className="bg-[#0F0F0F]/90 border-2 border-[#00FF91]/60 backdrop-blur-sm"
          style={{
            boxShadow: '0 0 20px rgba(0, 255, 145, 0.3)',
          }}
        >
          {/* Header */}
          <div className="border-b-2 border-[#00FF91]/40 p-6 text-center">
            <div className="text-[#00FF91] text-3xl font-mono font-bold mb-2 tracking-wider">
              SUMMIT.OS
            </div>
            <div className="text-[#006644] text-xs font-mono uppercase tracking-widest">
              Distributed Intelligence Fabric
            </div>
          </div>

          {/* Content */}
          <div className="p-8 space-y-6">
            <div className="space-y-2">
              <div className="text-[#00FF91] text-sm font-mono uppercase tracking-wider">
                System Access
              </div>
              <div className="text-[#006644] text-xs font-mono">
                Authenticate via OIDC to access tactical operations console
              </div>
            </div>

            {/* Status indicators */}
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-xs font-mono">
                <div className="w-2 h-2 bg-[#00FF91] animate-pulse" />
                <span className="text-[#00FF91]">Network: CONNECTED</span>
              </div>
              <div className="flex items-center gap-2 text-xs font-mono">
                <div className="w-2 h-2 bg-[#00FF91] animate-pulse" />
                <span className="text-[#00FF91]">Auth: READY</span>
              </div>
              <div className="flex items-center gap-2 text-xs font-mono">
                <div className="w-2 h-2 bg-[#FF9933] animate-pulse" />
                <span className="text-[#FF9933]">Status: AWAITING CREDENTIALS</span>
              </div>
            </div>

            {/* Login Button */}
            <button
              onClick={login}
              className="w-full bg-[#00FF91] text-[#0A0A0A] py-3 px-6 font-mono font-bold text-sm uppercase tracking-wider hover:bg-[#00CC74] transition-all duration-200 transform hover:scale-105"
              style={{
                boxShadow: '0 0 15px rgba(0, 255, 145, 0.5)',
              }}
            >
              ► AUTHENTICATE
            </button>

            {/* Footer note */}
            <div className="text-[#006644] text-[10px] font-mono text-center pt-4 border-t border-[#00FF91]/20">
              Secure OIDC Authentication
              <br />
              Multi-factor required for production systems
            </div>
          </div>
        </div>

        {/* Version info */}
        <div className="mt-4 text-center text-[#006644] text-[10px] font-mono">
          SUMMIT.OS v0.1.0 • Big Mountain Technologies
        </div>
      </div>
    </div>
  );
}
