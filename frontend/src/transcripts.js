import { PROJECT } from './scene3d.js';

/** The transcripts' vertex shader on the shared colour functions `colors`. `focus` {lit, alpha, grey, sourceAlpha,
 *  sourceGrey} (stage.js FOCUS_LIT, FOCUS_TX_ALPHA, FOCUS_TX_GREY, FOCUS_SOURCE_TX_ALPHA, FOCUS_SOURCE_TX_GREY): Go
 *  deeper's focus variant (2-D and 3-D, by each transcript's cell under it), where the transcripts of every cell but the
 *  uFocusLit ones go `grey` of the way to grey and fade toward `alpha` of theirs by uFocusDim (0-1), and those of the lit
 *  cells but the window's own (uFocusCell: the rows it names, second) `sourceGrey` and toward `sourceAlpha`; without it
 *  the shader is the plain one, unchanged. */
export function pointVertex(colors, focus = null) {
  const lit = focus ? `uniform uint uFocusLit[${focus.lit}];
uniform uint uFocusCell;
uniform float uFocusDim;
bool focusLit(uint id) {
  for (int k = 0; k < ${focus.lit}; k++) if (id == uFocusLit[k]) return id != 0u;
  return false;
}
` : '';
  // Grey as well as fainter: many faded dots on one spot still add up to the cell's colour.
  const dim = focus ? `  if (!focusLit(id)) {   // outside the shared colours
    vColor.rgb = mix(vColor.rgb, vec3(dot(vColor.rgb, vec3(0.2126, 0.7152, 0.0722))), ${(focus.grey ?? 0.85).toFixed(2)} * uFocusDim);
    vColor.a *= 1. - ${(1 - focus.alpha).toFixed(2)} * uFocusDim;
  } else if (id != uFocusCell) {   // the rows the window names: second to its cell
    vColor.rgb = mix(vColor.rgb, vec3(dot(vColor.rgb, vec3(0.2126, 0.7152, 0.0722))), ${(focus.sourceGrey ?? 0).toFixed(2)} * uFocusDim);
    vColor.a *= 1. - ${(1 - (focus.sourceAlpha ?? 1)).toFixed(2)} * uFocusDim;
  }
` : '';
  return `#version 300 es
precision highp float;
precision highp int;
precision highp usampler2DArray;
precision highp usampler2D;
in vec3 aPos;
uniform vec2 uCanvas;
uniform vec3 uView, uGeom;
uniform ivec3 uSize;
uniform vec2 uZ;
uniform int uLayer;
uniform float uTranscriptAlpha, uTranscriptSize;
uniform usampler2DArray uLayers;
uniform usampler2D uConsensus;
uniform bool uMarkOffFootprint, uHasLayers;
out float vRing;
out float vSize;
out vec4 vColor;
${PROJECT}
${colors}
${lit}void main() {
  // Records contain physical x/y and normalized z. Match the exported grid's
  // clamped z lookup. Tiled views awaiting a window use a subtle consensus tint.
  ivec2 ij = ivec2(floor((aPos.xy - uGeom.xy) / uGeom.z));
  int k = clamp(int(floor((aPos.z - uZ.x) / uZ.y)), 0, max(0, uSize.z - 1));
  uint id = 0u;
  if (all(greaterThanEqual(ij, ivec2(0))) && all(lessThan(ij, uSize.xy))) {
    id = uHasLayers ? texelFetch(uLayers, ivec3(ij, k), 0).r : texelFetch(uConsensus, ij, 0).r;
  }
  vColor = id == 0u ? vec4(0.55, 0.59, 0.64, 0.32) : vec4(displayColor(id), 1);
  if (!uHasLayers && id != 0u) vColor = vec4(mix(vec3(.55,.59,.64), displayColor(id), .45), .55);
  vec2 clip = (aPos.xy - uView.xy) * uView.z / uCanvas * 2.0;
  gl_Position = u3D ? project(aPos) : vec4(clip.x, -clip.y, 0, 1);
  vRing = 0.;
  if (!u3D && uHasLayers && id != 0u && uMarkOffFootprint) {
    vRing = texelFetch(uConsensus, ij, 0).r != id ? 1. : 0.;
  }
  gl_PointSize = uTranscriptSize + vRing * 4.;
  vColor.a *= uTranscriptAlpha;
  if (u3D) {
    float relativeDepth = max(gl_Position.w, .001) / max(uDistance, .001);
    float fade = max(.65, exp(-.25 * max(0., relativeDepth - 1.)));
    vColor = id == 0u ? vec4(.55,.59,.64,uTranscriptAlpha*.32*fade)
                      : vec4(displayColor(id),uTranscriptAlpha*fade);
    gl_PointSize = max(1.,uTranscriptSize * clamp(inversesqrt(relativeDepth),.85,1.5));
  }
  vSize = gl_PointSize;
  vColor.a *= clusterWeight(id);
${dim}  if (!u3D && uHasLayers && uLayer >= 0 && k != uLayer) gl_Position = vec4(2, 2, 2, 1);
}`;
}

export const POINT_FS = `#version 300 es
precision highp float;
in vec4 vColor;
in float vRing;
in float vSize;
out vec4 outColor;
void main() {
  float r = length(gl_PointCoord * 2.0 - 1.0);
  float alpha = 1.0 - smoothstep(0.05, 1.0, r);
  if (vRing > .5) {
    float core = max(.05, 1. - 4. / vSize);
    alpha = max(1. - smoothstep(0., core, r), (1. - smoothstep(1.-2./vSize, 1., r)) * smoothstep(max(core,1.-4./vSize),1.-2./vSize,r));
  }
  if (alpha < 0.02) discard;
  outColor = vec4(vColor.rgb, vColor.a * alpha);
}`;

// A cancellable run-scoped loader. main.js calls load() once the run's info.json exists: proseg writes the
// transcripts before info.json, so a 404 then means the run has none (runs made before transcripts were exported).
export class TranscriptLoader {
  constructor(stage) { this.stage = stage; }
  reset(run = null) {
    this.controller?.abort(); clearTimeout(this.timer);
    this.run = run; this.loaded = false; this.pending = false; this.missing = 0;
    this.stage.setTranscripts(null);
  }
  async load() {
    if (!this.run || this.loaded || this.pending) return;
    const run = this.run, controller = new AbortController();
    this.controller = controller; this.pending = true;
    try {
      const response = await fetch(`/api/v1/runs/${encodeURIComponent(run)}/transcripts`, { signal: controller.signal });
      if (response.status === 404) { this.missing += 1; return; }
      if (!response.ok) throw new Error(`Transcripts: HTTP ${response.status}`);
      const bytes = await response.arrayBuffer();
      if (bytes.byteLength % 16) throw new Error('Truncated transcript records');
      if (this.run !== run || controller.signal.aborted) return;
      this.stage.setTranscripts(bytes); this.loaded = true;
    } catch (error) {
      if (error.name !== 'AbortError') console.warn(error);
    } finally {
      if (this.controller === controller) {
        this.pending = false;
        if (!this.loaded && this.missing < 2) this.timer = setTimeout(() => this.load(), 2000);
      }
    }
  }
}
