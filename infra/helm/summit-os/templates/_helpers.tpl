{{/*
Expand the name of the chart.
*/}}
{{- define "summit-os.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "summit-os.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "summit-os.labels" -}}
helm.sh/chart: {{ include "summit-os.name" . }}-{{ .Chart.Version }}
{{ include "summit-os.selectorLabels" . }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "summit-os.selectorLabels" -}}
app.kubernetes.io/name: {{ include "summit-os.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Name of the ConfigMap.
*/}}
{{- define "summit-os.configmapName" -}}
{{- include "summit-os.fullname" . }}-config
{{- end }}

{{/*
Name of the Secret.
*/}}
{{- define "summit-os.secretName" -}}
{{- include "summit-os.fullname" . }}-secrets
{{- end }}

{{/*
Chart label helper.
*/}}
{{- define "summit-os.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}
