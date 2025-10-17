'use client'

import { HTMLAttributes } from 'react'

interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  variant?: string
}

export function Badge({ className = '', variant, ...props }: BadgeProps) {
  // variant prop is accepted for API compatibility; style as needed
  return <span data-variant={variant} className={`inline-flex items-center rounded px-2 py-0.5 text-xs border ${className}`} {...props} />
}
