"use client";

import React from "react";

type Toast = {
  id: string;
  title: string;
  description?: string;
  color?: string;
};

export default function PolicyNotifications() {
  const [toasts, setToasts] = React.useState<Toast[]>([]);

  React.useEffect(() => {
    function onDenied(e: Event) {
      const ce = e as CustomEvent;
      const { status, message, violations } = (ce.detail || {}) as any;
      const id = `${Date.now()}`;
      const desc = Array.isArray(violations) && violations.length
        ? violations.join("; ")
        : message || "Policy denied";
      setToasts((prev) => [
        ...prev,
        {
          id,
          title: `Policy Denied (${status || 403})`,
          description: desc,
          color: "#FF3333",
        },
      ]);
      // Auto-dismiss
      setTimeout(() => {
        setToasts((prev) => prev.filter((t) => t.id !== id));
      }, 6000);
    }
    window.addEventListener("policy-denied", onDenied as any);
    return () => window.removeEventListener("policy-denied", onDenied as any);
  }, []);

  if (!toasts.length) return null;

  return (
    <div className="fixed top-4 right-4 z-[10000] space-y-2">
      {toasts.map((t) => (
        <div
          key={t.id}
          className="w-96 p-3 border bg-[#0F0F0F]/95 backdrop-blur-sm"
          style={{
            borderColor: `${t.color || "#FF3333"}66`,
            boxShadow: `0 0 12px ${(t.color || "#FF3333")}33`,
          }}
        >
          <div className="flex items-start gap-2">
            <div
              className="w-2 h-2 mt-1 rounded-full"
              style={{ backgroundColor: t.color || "#FF3333", boxShadow: `0 0 6px ${(t.color || "#FF3333")}80` }}
            />
            <div className="flex-1">
              <div className="text-[#00FF91] font-mono text-sm">{t.title}</div>
              {t.description && (
                <div className="text-[#00CC74] font-mono text-xs mt-1">{t.description}</div>
              )}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
