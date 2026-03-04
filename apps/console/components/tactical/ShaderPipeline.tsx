'use client';

import React, { useEffect, useRef, useCallback } from 'react';

/**
 * ShaderPipeline — GPU post-processing effects for the tactical display.
 *
 * Wraps child content and applies fullscreen fragment shaders:
 *   NORMAL  — passthrough (no effect)
 *   NVG     — night vision green phosphor with grain + vignette
 *   FLIR    — thermal false-color palette
 *   CRT     — barrel distortion + scanlines + chromatic aberration + phosphor glow
 *
 * Implementation: captures the child content via a canvas overlay that reads
 * the rendered DOM as a WebGL texture using html2canvas-style frame capture
 * (actually uses requestAnimationFrame + drawImage from the child container).
 */

export type ShaderMode = 'NORMAL' | 'NVG' | 'FLIR' | 'CRT';

interface ShaderPipelineProps {
  mode: ShaderMode;
  children: React.ReactNode;
}

// ─── GLSL Shaders ──────────────────────────────────────────

const VERTEX_SHADER = `
  attribute vec2 a_position;
  attribute vec2 a_texCoord;
  varying vec2 v_texCoord;
  void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_texCoord = a_texCoord;
  }
`;

const PASSTHROUGH_FRAG = `
  precision mediump float;
  varying vec2 v_texCoord;
  uniform sampler2D u_texture;
  void main() {
    gl_FragColor = texture2D(u_texture, v_texCoord);
  }
`;

const NVG_FRAG = `
  precision mediump float;
  varying vec2 v_texCoord;
  uniform sampler2D u_texture;
  uniform float u_time;

  float rand(vec2 co) {
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
  }

  void main() {
    vec2 uv = v_texCoord;

    // Vignette
    float vignette = 1.0 - 0.4 * length(uv - 0.5);
    vignette = clamp(vignette, 0.0, 1.0);

    vec4 color = texture2D(u_texture, uv);

    // Convert to luminance
    float lum = dot(color.rgb, vec3(0.299, 0.587, 0.114));

    // Boost contrast
    lum = smoothstep(0.05, 0.85, lum);

    // Green phosphor tint
    vec3 nvg = vec3(lum * 0.15, lum * 1.0, lum * 0.2);

    // Grain noise
    float noise = rand(uv + u_time * 0.01) * 0.08;
    nvg += noise;

    // Phosphor bloom (brighten highlights)
    float bloom = smoothstep(0.6, 1.0, lum) * 0.3;
    nvg += vec3(bloom * 0.1, bloom * 0.6, bloom * 0.1);

    nvg *= vignette;

    gl_FragColor = vec4(nvg, color.a);
  }
`;

const FLIR_FRAG = `
  precision mediump float;
  varying vec2 v_texCoord;
  uniform sampler2D u_texture;
  uniform float u_time;

  float rand(vec2 co) {
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
  }

  vec3 thermalPalette(float t) {
    // Black → Blue → Magenta → Yellow → White
    if (t < 0.25) {
      float s = t / 0.25;
      return mix(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.6), s);
    } else if (t < 0.5) {
      float s = (t - 0.25) / 0.25;
      return mix(vec3(0.0, 0.0, 0.6), vec3(0.7, 0.0, 0.7), s);
    } else if (t < 0.75) {
      float s = (t - 0.5) / 0.25;
      return mix(vec3(0.7, 0.0, 0.7), vec3(1.0, 0.9, 0.0), s);
    } else {
      float s = (t - 0.75) / 0.25;
      return mix(vec3(1.0, 0.9, 0.0), vec3(1.0, 1.0, 1.0), s);
    }
  }

  void main() {
    vec2 uv = v_texCoord;
    vec4 color = texture2D(u_texture, uv);

    // Luminance
    float lum = dot(color.rgb, vec3(0.299, 0.587, 0.114));

    // Map through thermal palette
    vec3 thermal = thermalPalette(lum);

    // Temporal noise
    float noise = rand(uv + u_time * 0.005) * 0.04;
    thermal += noise;

    // Subtle vignette
    float vignette = 1.0 - 0.2 * length(uv - 0.5);
    thermal *= vignette;

    gl_FragColor = vec4(thermal, color.a);
  }
`;

const CRT_FRAG = `
  precision mediump float;
  varying vec2 v_texCoord;
  uniform sampler2D u_texture;
  uniform float u_time;
  uniform vec2 u_resolution;

  void main() {
    vec2 uv = v_texCoord;

    // Barrel distortion (CRT curvature)
    vec2 center = uv - 0.5;
    float r2 = dot(center, center);
    uv = uv + center * r2 * 0.12;

    // Check bounds after distortion
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
      gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
      return;
    }

    // Chromatic aberration (RGB sub-pixel separation)
    float aberration = 0.002;
    float r = texture2D(u_texture, uv + vec2(aberration, 0.0)).r;
    float g = texture2D(u_texture, uv).g;
    float b = texture2D(u_texture, uv - vec2(aberration, 0.0)).b;
    vec3 color = vec3(r, g, b);

    // Horizontal scanlines
    float scanline = sin(uv.y * u_resolution.y * 3.14159) * 0.5 + 0.5;
    scanline = mix(0.85, 1.0, scanline);
    color *= scanline;

    // Phosphor glow on bright elements
    float brightness = dot(color, vec3(0.299, 0.587, 0.114));
    float glow = smoothstep(0.5, 0.9, brightness) * 0.15;
    color += glow * vec3(0.1, 0.4, 0.1);

    // Vignette + corner shadow
    float dist = length(center) * 1.4;
    float vignette = 1.0 - dist * dist * 0.5;
    vignette = clamp(vignette, 0.0, 1.0);
    color *= vignette;

    // Subtle flicker
    float flicker = 1.0 - 0.02 * sin(u_time * 8.0);
    color *= flicker;

    gl_FragColor = vec4(color, 1.0);
  }
`;

// ─── Component ─────────────────────────────────────────────

export default function ShaderPipeline({ mode, children }: ShaderPipelineProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const glRef = useRef<WebGLRenderingContext | null>(null);
  const programRef = useRef<WebGLProgram | null>(null);
  const textureRef = useRef<WebGLTexture | null>(null);
  const rafRef = useRef<number>(0);
  const startTimeRef = useRef<number>(Date.now());
  const currentModeRef = useRef<ShaderMode>(mode);

  // Compile shader helper
  const compileShader = useCallback((gl: WebGLRenderingContext, type: number, source: string) => {
    const shader = gl.createShader(type);
    if (!shader) return null;
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      console.error('Shader compile error:', gl.getShaderInfoLog(shader));
      gl.deleteShader(shader);
      return null;
    }
    return shader;
  }, []);

  // Create program
  const createProgram = useCallback((gl: WebGLRenderingContext, fragSource: string) => {
    const vs = compileShader(gl, gl.VERTEX_SHADER, VERTEX_SHADER);
    const fs = compileShader(gl, gl.FRAGMENT_SHADER, fragSource);
    if (!vs || !fs) return null;

    const program = gl.createProgram();
    if (!program) return null;
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error('Program link error:', gl.getProgramInfoLog(program));
      gl.deleteProgram(program);
      return null;
    }

    // Clean up shaders
    gl.deleteShader(vs);
    gl.deleteShader(fs);

    return program;
  }, [compileShader]);

  const getFragShader = (m: ShaderMode) => {
    switch (m) {
      case 'NVG': return NVG_FRAG;
      case 'FLIR': return FLIR_FRAG;
      case 'CRT': return CRT_FRAG;
      default: return PASSTHROUGH_FRAG;
    }
  };

  // Initialize WebGL
  useEffect(() => {
    if (mode === 'NORMAL') return;

    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const gl = canvas.getContext('webgl', { premultipliedAlpha: false, alpha: true });
    if (!gl) {
      console.error('WebGL not available');
      return;
    }
    glRef.current = gl;

    // Create texture
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    textureRef.current = texture;

    // Fullscreen quad
    const positions = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
    const texCoords = new Float32Array([0, 1, 1, 1, 0, 0, 1, 0]);

    const posBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

    const texBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, texBuf);
    gl.bufferData(gl.ARRAY_BUFFER, texCoords, gl.STATIC_DRAW);

    // Build program for current mode
    const program = createProgram(gl, getFragShader(mode));
    if (!program) return;
    programRef.current = program;
    currentModeRef.current = mode;

    gl.useProgram(program);

    // Bind attributes
    const aPos = gl.getAttribLocation(program, 'a_position');
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

    const aTex = gl.getAttribLocation(program, 'a_texCoord');
    gl.bindBuffer(gl.ARRAY_BUFFER, texBuf);
    gl.enableVertexAttribArray(aTex);
    gl.vertexAttribPointer(aTex, 2, gl.FLOAT, false, 0, 0);

    startTimeRef.current = Date.now();

    // Create an offscreen canvas to capture the child content
    const captureCanvas = document.createElement('canvas');

    const render = () => {
      if (!glRef.current || !programRef.current || !containerRef.current) return;

      const rect = container.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const w = Math.floor(rect.width);
      const h = Math.floor(rect.height);

      if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
        canvas.width = w * dpr;
        canvas.height = h * dpr;
        canvas.style.width = `${w}px`;
        canvas.style.height = `${h}px`;
        captureCanvas.width = w;
        captureCanvas.height = h;
      }

      // Capture the underlying content using drawWindow fallback
      // Since we can't use drawWindow in standard browsers, we use
      // the fact that the content renders beneath the canvas overlay.
      // Instead, we capture the WebGL canvas from Cesium or the MapLibre canvas.
      const sourceCanvas = container.querySelector('canvas:not(.shader-overlay)') as HTMLCanvasElement;
      if (sourceCanvas) {
        try {
          const ctx = captureCanvas.getContext('2d');
          if (ctx) {
            ctx.drawImage(sourceCanvas, 0, 0, w, h);
          }

          gl.bindTexture(gl.TEXTURE_2D, textureRef.current);
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, captureCanvas);
        } catch {
          // Cross-origin or tainted canvas — skip frame
        }
      }

      gl.viewport(0, 0, canvas.width, canvas.height);

      // Set uniforms
      const uTime = gl.getUniformLocation(programRef.current, 'u_time');
      if (uTime) gl.uniform1f(uTime, (Date.now() - startTimeRef.current) / 1000);

      const uRes = gl.getUniformLocation(programRef.current, 'u_resolution');
      if (uRes) gl.uniform2f(uRes, canvas.width, canvas.height);

      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

      rafRef.current = requestAnimationFrame(render);
    };

    rafRef.current = requestAnimationFrame(render);

    return () => {
      cancelAnimationFrame(rafRef.current);
      if (programRef.current && glRef.current) {
        glRef.current.deleteProgram(programRef.current);
        programRef.current = null;
      }
      if (textureRef.current && glRef.current) {
        glRef.current.deleteTexture(textureRef.current);
        textureRef.current = null;
      }
      glRef.current = null;
    };
  }, [mode, createProgram]);

  const isActive = mode !== 'NORMAL';

  return (
    <div ref={containerRef} className="w-full h-full relative" data-shader-mode={mode}>
      {/* Child content (map/globe) renders here */}
      <div className="w-full h-full">
        {children}
      </div>

      {/* WebGL shader overlay canvas */}
      {isActive && (
        <canvas
          ref={canvasRef}
          className="shader-overlay absolute inset-0 pointer-events-none"
          style={{ zIndex: 5 }}
        />
      )}
    </div>
  );
}
