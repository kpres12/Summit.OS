'use client'

import { ReactNode, HTMLAttributes } from 'react'

export function Alert({ className = '', ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div role="alert" className={`rounded border p-3 ${className}`} {...props} />
}

export function AlertTitle({ className = '', ...props }: HTMLAttributes<HTMLHeadingElement>) {
  return <h5 className={`font-semibold mb-1 ${className}`} {...props} />
}

export function AlertDescription({ className = '', ...props }: HTMLAttributes<HTMLParagraphElement>) {
  return <p className={`text-sm text-muted-foreground ${className}`} {...props} />
}
