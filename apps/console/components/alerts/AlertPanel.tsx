'use client'

import { useState } from 'react'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { 
  AlertTriangle, 
  Flame, 
  Wind, 
  Shield, 
  Radio, 
  Navigation,
  Zap,
  CheckCircle,
  Clock,
  MapPin
} from 'lucide-react'
import { formatDistanceToNow } from 'date-fns'

interface AlertPanelProps {
  alerts: any[]
}

const severityIcons = {
  critical: Flame,
  high: AlertTriangle,
  medium: Wind,
  low: Shield
}

const severityColors = {
  critical: 'bg-red-500 text-white',
  high: 'bg-orange-500 text-white', 
  medium: 'bg-yellow-500 text-black',
  low: 'bg-blue-500 text-white'
}

const categoryIcons = {
  fire: Flame,
  smoke: Wind,
  weather: Wind,
  equipment: Radio,
  security: Shield,
  communication: Radio,
  navigation: Navigation,
  power: Zap,
  general: AlertTriangle
}

export function AlertPanel({ alerts }: AlertPanelProps) {
  const [selectedAlert, setSelectedAlert] = useState<any>(null)
  const [filter, setFilter] = useState('all')

  const filteredAlerts = alerts.filter(alert => {
    if (filter === 'all') return true
    return alert.severity === filter
  })

  const criticalAlerts = alerts.filter(alert => alert.severity === 'critical')
  const activeAlerts = alerts.filter(alert => alert.status === 'active')

  return (
    <div className="h-full flex flex-col">
      <Card className="rounded-none border-0 border-b">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-red-500" />
            Alerts
            <Badge variant="destructive" className="ml-auto">
              {criticalAlerts.length}
            </Badge>
          </CardTitle>
        </CardHeader>
      </Card>

      <div className="flex-1 overflow-hidden">
        <Tabs defaultValue="active" className="h-full flex flex-col">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="active">Active ({activeAlerts.length})</TabsTrigger>
            <TabsTrigger value="critical">Critical ({criticalAlerts.length})</TabsTrigger>
            <TabsTrigger value="all">All ({alerts.length})</TabsTrigger>
          </TabsList>

          <TabsContent value="active" className="flex-1 overflow-auto">
            <AlertList 
              alerts={activeAlerts} 
              selectedAlert={selectedAlert}
              onSelectAlert={setSelectedAlert}
            />
          </TabsContent>

          <TabsContent value="critical" className="flex-1 overflow-auto">
            <AlertList 
              alerts={criticalAlerts} 
              selectedAlert={selectedAlert}
              onSelectAlert={setSelectedAlert}
            />
          </TabsContent>

          <TabsContent value="all" className="flex-1 overflow-auto">
            <AlertList 
              alerts={filteredAlerts} 
              selectedAlert={selectedAlert}
              onSelectAlert={setSelectedAlert}
            />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}

function AlertList({ 
  alerts, 
  selectedAlert, 
  onSelectAlert 
}: { 
  alerts: any[]
  selectedAlert: any
  onSelectAlert: (alert: any) => void 
}) {
  if (alerts.length === 0) {
    return (
      <div className="p-4 text-center text-muted-foreground">
        No alerts
      </div>
    )
  }

  return (
    <div className="space-y-2 p-2">
      {alerts.map((alert) => {
        const SeverityIcon = severityIcons[alert.severity as keyof typeof severityIcons] || AlertTriangle
        const CategoryIcon = categoryIcons[alert.category as keyof typeof categoryIcons] || AlertTriangle
        
        return (
          <Card 
            key={alert.alert_id}
            className={`cursor-pointer transition-colors hover:bg-muted/50 ${
              selectedAlert?.alert_id === alert.alert_id ? 'ring-2 ring-primary' : ''
            }`}
            onClick={() => onSelectAlert(alert)}
          >
            <CardContent className="p-3">
              <div className="flex items-start gap-3">
                <div className={`p-2 rounded-full ${severityColors[alert.severity as keyof typeof severityColors]}`}>
                  <SeverityIcon className="h-4 w-4" />
                </div>
                
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <CategoryIcon className="h-4 w-4 text-muted-foreground" />
                    <span className="font-medium text-sm truncate">
                      {alert.title || alert.description}
                    </span>
                    <Badge 
                      variant={alert.severity === 'critical' ? 'destructive' : 'secondary'}
                      className="text-xs"
                    >
                      {alert.severity}
                    </Badge>
                  </div>
                  
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <MapPin className="h-3 w-3" />
                    <span>
                      {alert.location?.latitude?.toFixed(4)}, {alert.location?.longitude?.toFixed(4)}
                    </span>
                    <Clock className="h-3 w-3 ml-2" />
                    <span>
                      {formatDistanceToNow(new Date(alert.timestamp), { addSuffix: true })}
                    </span>
                  </div>
                  
                  {alert.source && (
                    <div className="text-xs text-muted-foreground mt-1">
                      Source: {alert.source}
                    </div>
                  )}
                </div>
                
                {alert.status === 'acknowledged' && (
                  <CheckCircle className="h-4 w-4 text-green-500" />
                )}
              </div>
            </CardContent>
          </Card>
        )
      })}
    </div>
  )
}
