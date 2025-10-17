'use client'

import { HTMLAttributes } from 'react'

interface TabsProps extends HTMLAttributes<HTMLDivElement> {
  defaultValue?: string
}

export function Tabs({ className = '', defaultValue, ...props }: TabsProps) {
  return <div data-default-value={defaultValue} className={`flex flex-col ${className}`} {...props} />
}

interface TabsListProps extends HTMLAttributes<HTMLDivElement> {}
export function TabsList({ className = '', ...props }: TabsListProps) {
  return <div className={`flex gap-2 border-b ${className}`} {...props} />
}

interface TabsTriggerProps extends HTMLAttributes<HTMLButtonElement> {
  value?: string
}
export function TabsTrigger({ className = '', value, ...props }: TabsTriggerProps) {
  return <button data-value={value} className={`px-3 py-1 text-sm ${className}`} {...props} />
}

interface TabsContentProps extends HTMLAttributes<HTMLDivElement> {
  value?: string
}
export function TabsContent({ className = '', value, ...props }: TabsContentProps) {
  return <div data-value={value} className={`flex-1 ${className}`} {...props} />
}
