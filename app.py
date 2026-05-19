#!/usr/bin/env python3
"""
GLB Desktop Viewer — hand gesture control mirrors the web version exactly.

Gesture theory (identical to app.py):
  1 hand + pinch + drag  → rotate  (accumulated deltas × ROTATION_SENSITIVITY,
                                     smoothed with ROTATION_SMOOTHING lerp,
                                     then a second lerp in the render tick)
  2 hands + both pinch   → zoom    (index-tip distance delta × ZOOM_SENSITIVITY,
                                     smoothed with ZOOM_SMOOTHING lerp,
                                     then a second lerp in the render tick)
  Pinch hysteresis       → PINCH_ENGAGE 0.07 / PINCH_RELEASE 0.10
  PINCH_LOST_TOLERANCE   → 3 frames before resetting pinch position
  Gesture expiry         → 0.5 s (same as web GESTURE_EXPIRY_SECONDS)

Camera / rotation (same as Three.js):
  Camera orbits on a sphere around the model (model fixed at origin).
  theta = horizontal angle (left/right), phi = vertical angle (up/down)
  zoom changes camera distance, lerped with ZOOM_LERP = 0.15

Controls (mouse fallback):
  LMB drag = rotate model  |  Scroll = zoom  |  R = reset  |  Q = quit
"""

# ─── Config ───────────────────────────────────────────────────────────────────
USE_HAND_GESTURES = True     # set False to use mouse only
SHOW_GESTURE_CAM  = True     # show camera window when gestures are enabled
WINDOW_WIDTH      = 1280
WINDOW_HEIGHT     = 720
BACKGROUND_COLOR  = (0.08, 0.08, 0.12, 1.0)

# ── Web-version gesture constants (copy-pasted from app.py) ───────────────────
PINCH_ENGAGE           = 0.07
PINCH_RELEASE          = 0.10
ZOOM_SENSITIVITY       = 2.5
ZOOM_SMOOTHING         = 0.2        # first lerp (Python/gesture thread)
ZOOM_LERP              = 0.15       # second lerp (render tick)
ROTATION_SENSITIVITY   = 180        # degrees per unit of normalised delta
ROTATION_SMOOTHING     = 0.35       # first lerp (gesture thread)
ROT_LERP               = 0.2        # second lerp (render tick)
PINCH_LOST_TOLERANCE   = 3          # frames before resetting pinch position
GESTURE_EXPIRY         = 0.5        # seconds
# ─────────────────────────────────────────────────────────────────────────────

import sys, os, math, glob, threading, ctypes, time

def pip_install(*pkgs):
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", *pkgs, "-q"])

try:
    import numpy as np
except ImportError:
    print("Installing numpy..."); pip_install("numpy"); import numpy as np

try:
    import trimesh
except ImportError:
    print("Installing trimesh..."); pip_install("trimesh[easy]"); import trimesh

try:
    import pyglet
    from pyglet.gl import *
except ImportError:
    print("Installing pyglet..."); pip_install("pyglet"); import pyglet; from pyglet.gl import *

try:
    import tkinter as tk
    from tkinter import filedialog
    HAS_TK = True
except ImportError:
    HAS_TK = False

HAS_GESTURE = False
if USE_HAND_GESTURES:
    try:
        import cv2
        import mediapipe as mp
        HAS_GESTURE = True
    except ImportError:
        try:
            pip_install("opencv-python", "mediapipe")
            import cv2
            import mediapipe as mp
            HAS_GESTURE = True
        except Exception as e:
            print(f"[Gesture] Could not import cv2/mediapipe: {e} — mouse only")


# ─────────────────────────────────────────────────────────────────────────────
#  Shared gesture state  (gesture thread writes, render tick reads)
# ─────────────────────────────────────────────────────────────────────────────
_gesture_lock  = threading.Lock()
_gesture_state = {
    "type":       None,   # "zoom" | "rotation" | None
    "zoom_scale": 1.0,    # smoothed_zoom_scale (first lerp already applied)
    "rotation_x": 0.0,    # smoothed_rotation["x"] in DEGREES (first lerp applied)
    "rotation_y": 0.0,
    "timestamp":  0.0,
}


# ─────────────────────────────────────────────────────────────────────────────
#  Gesture thread — exact port of generate_camera() logic from app.py
# ─────────────────────────────────────────────────────────────────────────────
def run_gesture_thread():
    """
    Mirrors generate_camera() in app.py exactly:
      - Hysteresis pinch detect
      - 2-hand zoom with ZOOM_SENSITIVITY + first ZOOM_SMOOTHING lerp
      - 1-hand pinch-drag rotation with ROTATION_SENSITIVITY + first ROTATION_SMOOTHING lerp
      - PINCH_LOST_TOLERANCE before resetting previous_pinch_position
      - Gesture expiry via timestamp
    """
    mp_hands = mp.solutions.hands
    detector = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    # ── same state variables as app.py ────────────────────────────────────────
    hand_pinch_state       = [False, False]
    previous_hand_distance = None
    zoom_scale             = 1.0
    smoothed_zoom_scale    = 1.0

    raw_rotation           = {"x": 0.0, "y": 0.0}
    smoothed_rotation      = {"x": 0.0, "y": 0.0}

    previous_pinch_position = None
    pinch_lost_frames       = 0

    # ── helpers ───────────────────────────────────────────────────────────────
    def _lerp(cur, tgt, f):
        return cur + (tgt - cur) * f

    def is_pinching_with_hysteresis(lms, hand_idx):
        """Exact copy of app.py is_pinching_with_hysteresis()."""
        thumb = lms.landmark[4]
        index = lms.landmark[8]
        dist  = math.hypot(thumb.x - index.x, thumb.y - index.y)
        if hand_pinch_state[hand_idx]:
            if dist > PINCH_RELEASE:
                hand_pinch_state[hand_idx] = False
        else:
            if dist < PINCH_ENGAGE:
                hand_pinch_state[hand_idx] = True
        return hand_pinch_state[hand_idx]

    def calculate_hand_distance(h1, h2):
        """Distance between index fingertips (landmark 8) — same as app.py."""
        x1, y1 = h1.landmark[8].x, h1.landmark[8].y
        x2, y2 = h2.landmark[8].x, h2.landmark[8].y
        return math.hypot(x2 - x1, y2 - y1)

    def get_pinch_center(lms):
        """Midpoint of thumb tip + index tip — same as app.py."""
        t, i = lms.landmark[4], lms.landmark[8]
        return (t.x + i.x) / 2, (t.y + i.y) / 2

    # ── open camera ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Gesture] Could not open camera — gesture disabled")
        return

    print("[Gesture] Camera open — hand gesture active")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame   = cv2.flip(frame, 1)
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)
        now     = time.time()

        zoom_gesture_data = None
        rot_gesture_data  = None

        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)

            if SHOW_GESTURE_CAM:
                for lms in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, lms, mp_hands.HAND_CONNECTIONS
                    )

            # ── 2 hands → ZOOM ────────────────────────────────────────────────
            if num_hands == 2:
                hand0 = results.multi_hand_landmarks[0]
                hand1 = results.multi_hand_landmarks[1]
                p0    = is_pinching_with_hysteresis(hand0, 0)
                p1    = is_pinching_with_hysteresis(hand1, 1)

                if p0 and p1:
                    cur_dist = calculate_hand_distance(hand0, hand1)
                    if previous_hand_distance is not None:
                        delta       = cur_dist - previous_hand_distance
                        zoom_scale += delta * ZOOM_SENSITIVITY
                        zoom_scale  = max(0.5, min(3.0, zoom_scale))
                    # first smoothing lerp (ZOOM_SMOOTHING = 0.2)
                    smoothed_zoom_scale = _lerp(smoothed_zoom_scale, zoom_scale, ZOOM_SMOOTHING)
                    previous_hand_distance = cur_dist
                    zoom_gesture_data = {"type": "zoom", "zoom_scale": smoothed_zoom_scale}

                    label = "ZOOM IN +" if zoom_scale > smoothed_zoom_scale else "ZOOM OUT -"
                    if SHOW_GESTURE_CAM:
                        cv2.putText(frame, f"{label}  {smoothed_zoom_scale:.2f}x",
                                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    previous_hand_distance = None

                # reset rotation tracking when 2 hands are used
                previous_pinch_position = None
                pinch_lost_frames = 0

            # ── 1 hand → ROTATION ─────────────────────────────────────────────
            elif num_hands == 1:
                handLms  = results.multi_hand_landmarks[0]
                pinching = is_pinching_with_hysteresis(handLms, 0)

                if pinching:
                    pinch_lost_frames = 0
                    px, py = get_pinch_center(handLms)

                    if previous_pinch_position is not None:
                        dx = px - previous_pinch_position[0]
                        dy = py - previous_pinch_position[1]
                        # accumulate raw degrees — same as app.py
                        raw_rotation["x"] += dy * ROTATION_SENSITIVITY
                        raw_rotation["y"] += dx * ROTATION_SENSITIVITY

                    # first smoothing lerp (ROTATION_SMOOTHING = 0.35)
                    smoothed_rotation["x"] = _lerp(
                        smoothed_rotation["x"], raw_rotation["x"], ROTATION_SMOOTHING
                    )
                    smoothed_rotation["y"] = _lerp(
                        smoothed_rotation["y"], raw_rotation["y"], ROTATION_SMOOTHING
                    )

                    rot_gesture_data = {
                        "type":       "rotation",
                        "rotation_x": smoothed_rotation["x"],   # degrees
                        "rotation_y": smoothed_rotation["y"],
                    }

                    if SHOW_GESTURE_CAM:
                        h, w, _ = frame.shape
                        cv2.circle(frame, (int(px * w), int(py * h)), 15, (255, 0, 255), 3)
                        cv2.putText(frame, "ROTATE (Pinch+Drag)",
                                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

                    previous_pinch_position = (px, py)

                else:
                    # tolerance before resetting position — same as app.py
                    pinch_lost_frames += 1
                    if pinch_lost_frames >= PINCH_LOST_TOLERANCE:
                        previous_pinch_position = None
                    if SHOW_GESTURE_CAM:
                        cv2.putText(frame, "Pinch+drag to rotate",
                                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

                previous_hand_distance = None

        else:
            # no hands → reset all tracking
            previous_hand_distance  = None
            previous_pinch_position = None
            pinch_lost_frames       = 0
            hand_pinch_state[0]     = False
            hand_pinch_state[1]     = False

        # ── push to shared state ──────────────────────────────────────────────
        with _gesture_lock:
            if zoom_gesture_data:
                _gesture_state.update(zoom_gesture_data)
                _gesture_state["timestamp"] = now
            elif rot_gesture_data:
                _gesture_state.update(rot_gesture_data)
                _gesture_state["timestamp"] = now
            elif now - _gesture_state["timestamp"] > GESTURE_EXPIRY:
                _gesture_state["type"] = None

        if SHOW_GESTURE_CAM:
            cv2.imshow("Gesture Cam (Q to close cam)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
#  Math helpers  (Three.js conventions)
# ─────────────────────────────────────────────────────────────────────────────
def look_at(eye, target, up):
    f = target - eye;  f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = r;  m[1, :3] = u;  m[2, :3] = -f
    t = np.eye(4, dtype=np.float32)
    t[:3, 3] = -eye
    return m @ t

def perspective(fov_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fov_deg) / 2)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1
    return m

def rotation_x(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]], dtype=np.float32)

def rotation_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Mesh → OpenGL VBO
# ─────────────────────────────────────────────────────────────────────────────
class GLMesh:
    def __init__(self, vertices, normals, colors, faces):
        self.vertex_count = len(faces) * 3
        if self.vertex_count == 0:
            return
        idx  = faces.flatten()
        v    = vertices[idx].astype(np.float32)
        n    = normals[idx].astype(np.float32) if normals is not None else np.zeros_like(v)
        c    = colors[idx].astype(np.float32)  if colors  is not None else np.ones((len(v), 4), np.float32)
        data = np.hstack([v, n, c]).astype(np.float32)
        self.vao = GLuint(); glGenVertexArrays(1, self.vao)
        self.vbo = GLuint(); glGenBuffers(1, self.vbo)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data.ctypes.data, GL_STATIC_DRAW)
        stride = 10 * 4
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))
        glBindVertexArray(0)

    def draw(self):
        if self.vertex_count == 0: return
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
        glBindVertexArray(0)

    def delete(self):
        glDeleteVertexArrays(1, self.vao)
        glDeleteBuffers(1, self.vbo)


# ─────────────────────────────────────────────────────────────────────────────
#  GLSL shaders
# ─────────────────────────────────────────────────────────────────────────────
VERT_SRC = b"""
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNorm;
layout(location=2) in vec4 aColor;
uniform mat4 uMVP;
uniform mat4 uModel;
out vec3 vNormal;
out vec3 vFragPos;
out vec4 vColor;
out vec3 vWorldPos;
void main(){
    vec4 worldPos = uModel * vec4(aPos, 1.0);
    vFragPos  = worldPos.xyz;
    vWorldPos = worldPos.xyz;
    vNormal   = mat3(transpose(inverse(uModel))) * aNorm;
    vColor    = aColor;
    gl_Position = uMVP * vec4(aPos, 1.0);
}
"""

FRAG_SRC = b"""
#version 330 core
in vec3 vNormal;
in vec3 vFragPos;
in vec4 vColor;
in vec3 vWorldPos;
out vec4 FragColor;
uniform vec3 uLightDir;
uniform vec3 uLightDir2;
uniform vec3 uCamPos;
uniform float uTime;

void main(){
    vec3 norm    = normalize(vNormal);
    vec3 viewDir = normalize(uCamPos - vFragPos);

    // --- Base Phong lighting (unchanged) ---
    vec3 ambient  = 0.35 * vColor.rgb;
    float diff1   = max(dot(norm, normalize(uLightDir)),  0.0);
    float diff2   = max(dot(norm, normalize(uLightDir2)), 0.0) * 0.5;
    vec3 diffuse  = (diff1 + diff2) * vColor.rgb;
    vec3 reflDir  = reflect(-normalize(uLightDir), norm);
    float spec    = pow(max(dot(viewDir, reflDir), 0.0), 48.0);
    vec3 specular = 0.5 * spec * vec3(1.0, 0.95, 0.7);   // warm gold highlight
    vec3 baseColor = ambient + diffuse + specular;

    // --- Fresnel rim glow (cyan/blue edge light) ---
    float fresnel = pow(1.0 - clamp(dot(norm, viewDir), 0.0, 1.0), 3.5);
    vec3 rimColor = vec3(0.0, 0.75, 1.0) * fresnel * 2.2;

    // --- Scanlines (horizontal bands scrolling upward) ---
    float scanSpeed   = uTime * 1.8;
    float scanFreq    = 60.0;
    float scanLine    = sin(vWorldPos.y * scanFreq + scanSpeed);
    float scanBright  = 0.5 + 0.5 * scanLine;          // 0..1
    float scanEffect  = mix(1.0, scanBright, 0.06);     // subtle: 6% modulation

    // --- Fine interference lines (secondary, faster) ---
    float fine = sin(vWorldPos.y * 200.0 - uTime * 4.0) * 0.5 + 0.5;
    float fineEffect = mix(1.0, fine, 0.025);

    // --- Hologram flicker (very subtle global pulse) ---
    float flicker = 0.96 + 0.04 * sin(uTime * 23.7) * sin(uTime * 7.3);

    // --- Combine ---
    vec3 holo = (baseColor * scanEffect * fineEffect + rimColor) * flicker;

    // Alpha: rim edges slightly more transparent for ghost feel
    float alpha = vColor.a * mix(0.88, 1.0, 1.0 - fresnel * 0.5);

    FragColor = vec4(holo, alpha);
}
"""

def _compile_shader(src, kind):
    sh  = glCreateShader(kind)
    ptr = ctypes.cast(
        ctypes.pointer(ctypes.c_char_p(src)),
        ctypes.POINTER(ctypes.POINTER(GLchar))
    )
    glShaderSource(sh, 1, ptr, None)
    glCompileShader(sh)
    ok = GLint()
    glGetShaderiv(sh, GL_COMPILE_STATUS, ok)
    if not ok.value:
        buf = ctypes.create_string_buffer(1024)
        glGetShaderInfoLog(sh, 1024, None, buf)
        raise RuntimeError(f"Shader error: {buf.value.decode()}")
    return sh

def build_program():
    vs = _compile_shader(VERT_SRC, GL_VERTEX_SHADER)
    fs = _compile_shader(FRAG_SRC, GL_FRAGMENT_SHADER)
    p  = glCreateProgram()
    glAttachShader(p, vs); glAttachShader(p, fs)
    glLinkProgram(p)
    glDeleteShader(vs); glDeleteShader(fs)
    return p

def ul(prog, name):
    return glGetUniformLocation(prog, name.encode())


# ─────────────────────────────────────────────────────────────────────────────
#  Star Field  (drawn before model, no depth write)
# ─────────────────────────────────────────────────────────────────────────────
STAR_VERT_SRC = b"""
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in float aBrightness;
uniform mat4 uVP;
uniform float uTime;
out float vBrightness;
out float vTwinkle;
void main(){
    gl_Position  = uVP * vec4(aPos, 1.0);
    gl_PointSize = 1.5 + aBrightness * 2.5;
    // each star twinkles at its own frequency using its x position as seed
    float seed   = aPos.x * 127.3 + aPos.z * 311.7;
    vTwinkle     = 0.55 + 0.45 * sin(uTime * 2.1 + seed);
    vBrightness  = aBrightness;
}
"""

STAR_FRAG_SRC = b"""
#version 330 core
in float vBrightness;
in float vTwinkle;
out vec4 FragColor;
void main(){
    // soft circular point
    vec2  c    = gl_PointCoord - 0.5;
    float dist = dot(c, c) * 4.0;          // 0 at centre, 1 at edge
    float alpha = (1.0 - dist) * vBrightness * vTwinkle;
    if (alpha < 0.01) discard;
    // colour: faint blue-white, brighter stars slightly warmer
    vec3 cold = vec3(0.6, 0.75, 1.0);
    vec3 warm = vec3(1.0, 0.95, 0.85);
    vec3 col  = mix(cold, warm, vBrightness * 0.4);
    FragColor = vec4(col, alpha);
}
"""

class StarField:
    NUM_STARS  = 350
    RADIUS     = 800.0   # place stars on a large sphere around the scene

    def __init__(self):
        self.prog = self._build_prog()
        rng = np.random.default_rng(42)

        # random points on a sphere (fibonacci-ish via uniform sampling)
        phi   = np.arccos(1.0 - 2.0 * rng.random(self.NUM_STARS))
        theta = 2.0 * math.pi * rng.random(self.NUM_STARS)
        r     = self.RADIUS
        xs    = (r * np.sin(phi) * np.cos(theta)).astype(np.float32)
        ys    = (r * np.sin(phi) * np.sin(theta)).astype(np.float32)
        zs    = (r * np.cos(phi)).astype(np.float32)
        bright = rng.random(self.NUM_STARS).astype(np.float32) ** 0.5  # skew bright

        data = np.column_stack([xs, ys, zs, bright]).astype(np.float32)

        self.vao = GLuint(); glGenVertexArrays(1, self.vao)
        self.vbo = GLuint(); glGenBuffers(1, self.vbo)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data.ctypes.data, GL_STATIC_DRAW)
        stride = 4 * 4
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glBindVertexArray(0)
        self.count = self.NUM_STARS

    def _build_prog(self):
        vs = _compile_shader(STAR_VERT_SRC, GL_VERTEX_SHADER)
        fs = _compile_shader(STAR_FRAG_SRC, GL_FRAGMENT_SHADER)
        p  = glCreateProgram()
        glAttachShader(p, vs); glAttachShader(p, fs)
        glLinkProgram(p)
        glDeleteShader(vs); glDeleteShader(fs)
        return p

    def draw(self, view, proj, t):
        vp = (proj @ view).T.astype(np.float32)

        def fptr(m):
            return m.flatten().ctypes.data_as(ctypes.POINTER(GLfloat))

        glUseProgram(self.prog)
        glUniformMatrix4fv(glGetUniformLocation(self.prog, b"uVP"), 1, GL_FALSE, fptr(vp))
        glUniform1f(glGetUniformLocation(self.prog, b"uTime"), t)

        glEnable(GL_PROGRAM_POINT_SIZE)
        glDepthMask(GL_FALSE)           # don't write to depth buffer
        glBindVertexArray(self.vao)
        glDrawArrays(GL_POINTS, 0, self.count)
        glBindVertexArray(0)
        glDepthMask(GL_TRUE)
        glDisable(GL_PROGRAM_POINT_SIZE)
        glUseProgram(0)

    def delete(self):
        glDeleteVertexArrays(1, self.vao)
        glDeleteBuffers(1, self.vbo)


# ─────────────────────────────────────────────────────────────────────────────
#  Ground Glow Ring  (flat ellipse beneath the model, additive blend)
# ─────────────────────────────────────────────────────────────────────────────
RING_VERT_SRC = b"""
#version 330 core
layout(location=0) in vec2 aUV;       // -1..1 on the disc
uniform mat4 uMVP;
uniform float uRadius;
uniform float uYOffset;
out vec2 vUV;
void main(){
    vUV = aUV;
    vec3 pos = vec3(aUV.x * uRadius, uYOffset, aUV.y * uRadius);
    gl_Position = uMVP * vec4(pos, 1.0);
}
"""

RING_FRAG_SRC = b"""
#version 330 core
in vec2 vUV;
out vec4 FragColor;
uniform float uTime;
uniform vec3  uInnerColor;
uniform vec3  uOuterColor;
void main(){
    float r      = length(vUV);              // 0 = centre, 1 = edge
    if (r > 1.0) discard;

    // ring shape: bright band between 0.55 and 0.95, fading at both edges
    float inner  = smoothstep(0.45, 0.65, r);
    float outer  = 1.0 - smoothstep(0.80, 1.00, r);
    float band   = inner * outer;

    // slow rotation pulse on top of band
    float angle  = atan(vUV.y, vUV.x);
    float pulse  = 0.75 + 0.25 * sin(angle * 6.0 - uTime * 1.4);

    // soft overall breathing
    float breath = 0.80 + 0.20 * sin(uTime * 1.1);

    vec3  col    = mix(uInnerColor, uOuterColor, r);
    float alpha  = band * pulse * breath * 0.55;
    FragColor    = vec4(col, alpha);
}
"""

class GlowRing:
    SEGMENTS = 128

    def __init__(self, radius=1.0, y_offset=0.0,
                 inner_color=(0.0, 0.6, 1.0),
                 outer_color=(0.4, 0.0, 0.8)):
        self.radius      = radius
        self.y_offset    = y_offset
        self.inner_color = inner_color
        self.outer_color = outer_color
        self.prog        = self._build_prog()

        # build a filled disc via triangle fan  (centre + SEGMENTS rim points)
        N    = self.SEGMENTS
        uvs  = [(0.0, 0.0)]                                  # centre
        for i in range(N + 1):
            a = 2.0 * math.pi * i / N
            uvs.append((math.cos(a), math.sin(a)))
        data = np.array(uvs, dtype=np.float32)

        self.vao = GLuint(); glGenVertexArrays(1, self.vao)
        self.vbo = GLuint(); glGenBuffers(1, self.vbo)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data.ctypes.data, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
        glBindVertexArray(0)
        self.count = N + 2   # centre + N+1 rim

    def _build_prog(self):
        vs = _compile_shader(RING_VERT_SRC, GL_VERTEX_SHADER)
        fs = _compile_shader(RING_FRAG_SRC, GL_FRAGMENT_SHADER)
        p  = glCreateProgram()
        glAttachShader(p, vs); glAttachShader(p, fs)
        glLinkProgram(p)
        glDeleteShader(vs); glDeleteShader(fs)
        return p

    def draw(self, mvp, t):
        def fptr(m):
            return m.flatten().ctypes.data_as(ctypes.POINTER(GLfloat))

        glUseProgram(self.prog)
        glUniformMatrix4fv(glGetUniformLocation(self.prog, b"uMVP"),    1, GL_FALSE, fptr(mvp))
        glUniform1f(glGetUniformLocation(self.prog, b"uRadius"),        self.radius)
        glUniform1f(glGetUniformLocation(self.prog, b"uYOffset"),       self.y_offset)
        glUniform1f(glGetUniformLocation(self.prog, b"uTime"),          t)
        glUniform3f(glGetUniformLocation(self.prog, b"uInnerColor"),   *self.inner_color)
        glUniform3f(glGetUniformLocation(self.prog, b"uOuterColor"),   *self.outer_color)

        # additive blending so the ring brightens whatever is behind it
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        glDepthMask(GL_FALSE)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLE_FAN, 0, self.count)
        glBindVertexArray(0)
        glDepthMask(GL_TRUE)
        # restore normal alpha blend
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glUseProgram(0)

    def delete(self):
        glDeleteVertexArrays(1, self.vao)
        glDeleteBuffers(1, self.vbo)


# ─────────────────────────────────────────────────────────────────────────────
#  Floating Dust Particles  (rise around the model, additive blend)
# ─────────────────────────────────────────────────────────────────────────────
PARTICLE_VERT_SRC = b"""
#version 330 core
layout(location=0) in vec3  aBasePos;   // spawn position (static)
layout(location=1) in float aOffset;    // time offset per particle (0..1)
layout(location=2) in float aSpeed;     // rise speed multiplier
layout(location=3) in float aSize;      // base point size
layout(location=4) in float aSwirl;     // horizontal swirl radius
layout(location=5) in float aSwirlFreq; // swirl frequency
uniform mat4  uVP;
uniform float uTime;
uniform float uHeight;   // full travel height before respawn
out float vAlpha;
out float vColor;        // 0=cyan, 1=gold  per particle
void main(){
    // phase: 0 -> 1 over the particle's lifetime, then loops
    float phase = fract((uTime * aSpeed + aOffset));

    // fade in first 15%, full 15-75%, fade out last 25%
    float fadeIn  = smoothstep(0.0,  0.15, phase);
    float fadeOut = 1.0 - smoothstep(0.75, 1.0,  phase);
    vAlpha = fadeIn * fadeOut * 0.75;

    // rise straight up
    float y = aBasePos.y + phase * uHeight;

    // gentle horizontal swirl (Lissajous drift)
    float t  = uTime * aSwirlFreq + aOffset * 6.28318;
    float dx = aSwirl * sin(t);
    float dz = aSwirl * cos(t * 0.7 + 1.3);

    vec3 pos = vec3(aBasePos.x + dx, y, aBasePos.z + dz);
    gl_Position  = uVP * vec4(pos, 1.0);
    gl_PointSize = aSize * (1.0 - phase * 0.5);   // shrink as it rises

    // use aOffset as colour seed
    vColor = fract(aOffset * 7.3);
}
"""

PARTICLE_FRAG_SRC = b"""
#version 330 core
in float vAlpha;
in float vColor;
out vec4 FragColor;
void main(){
    // round soft point
    vec2  c    = gl_PointCoord - 0.5;
    float dist = dot(c, c) * 4.0;
    if (dist > 1.0) discard;
    float soft = 1.0 - dist;

    // cyan <-> gold palette based on per-particle seed
    vec3 cyan = vec3(0.1, 0.8, 1.0);
    vec3 gold = vec3(1.0, 0.85, 0.2);
    vec3 col  = mix(cyan, gold, step(0.5, vColor));

    FragColor = vec4(col, soft * vAlpha);
}
"""

class ParticleSystem:
    NUM = 220

    def __init__(self, spawn_radius=1.0, height=1.0):
        self.prog   = self._build_prog()
        self.height = height
        rng = np.random.default_rng(7)

        N = self.NUM
        # spawn ring: random angle, random radius fraction
        angles   = rng.random(N) * 2.0 * math.pi
        radii    = (0.3 + 0.7 * rng.random(N)) * spawn_radius
        base_x   = (radii * np.cos(angles)).astype(np.float32)
        base_z   = (radii * np.sin(angles)).astype(np.float32)
        base_y   = (rng.random(N) * height * -0.3).astype(np.float32)  # stagger start heights

        offsets   = rng.random(N).astype(np.float32)
        speeds    = (0.03 + 0.07 * rng.random(N)).astype(np.float32)
        sizes     = (2.5  + 3.5  * rng.random(N)).astype(np.float32)
        swirls    = (0.04 + 0.12 * rng.random(N) * spawn_radius).astype(np.float32)
        swirl_f   = (0.3  + 0.7  * rng.random(N)).astype(np.float32)

        data = np.column_stack([
            base_x, base_y, base_z,   # aBasePos  (3)
            offsets,                   # aOffset   (1)
            speeds,                    # aSpeed    (1)
            sizes,                     # aSize     (1)
            swirls,                    # aSwirl    (1)
            swirl_f,                   # aSwirlFreq(1)
        ]).astype(np.float32)          # 9 floats per particle

        self.vao = GLuint(); glGenVertexArrays(1, self.vao)
        self.vbo = GLuint(); glGenBuffers(1, self.vbo)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data.ctypes.data, GL_STATIC_DRAW)
        stride = 9 * 4
        for idx, (offset, size) in enumerate([
            (0, 3), (3*4, 1), (4*4, 1), (5*4, 1), (6*4, 1), (7*4, 1)
        ]):
            glEnableVertexAttribArray(idx)
            glVertexAttribPointer(idx, size, GL_FLOAT, GL_FALSE,
                                  stride, ctypes.c_void_p(offset))
        glBindVertexArray(0)

    def _build_prog(self):
        vs = _compile_shader(PARTICLE_VERT_SRC, GL_VERTEX_SHADER)
        fs = _compile_shader(PARTICLE_FRAG_SRC, GL_FRAGMENT_SHADER)
        p  = glCreateProgram()
        glAttachShader(p, vs); glAttachShader(p, fs)
        glLinkProgram(p)
        glDeleteShader(vs); glDeleteShader(fs)
        return p

    def draw(self, view, proj, t):
        vp = (proj @ view).T.astype(np.float32)

        def fptr(m):
            return m.flatten().ctypes.data_as(ctypes.POINTER(GLfloat))

        glUseProgram(self.prog)
        glUniformMatrix4fv(glGetUniformLocation(self.prog, b"uVP"),     1, GL_FALSE, fptr(vp))
        glUniform1f(glGetUniformLocation(self.prog, b"uTime"),          t)
        glUniform1f(glGetUniformLocation(self.prog, b"uHeight"),        self.height)

        glEnable(GL_PROGRAM_POINT_SIZE)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)   # additive so they glow
        glDepthMask(GL_FALSE)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_POINTS, 0, self.NUM)
        glBindVertexArray(0)
        glDepthMask(GL_TRUE)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_PROGRAM_POINT_SIZE)
        glUseProgram(0)

    def delete(self):
        glDeleteVertexArrays(1, self.vao)
        glDeleteBuffers(1, self.vbo)


# ─────────────────────────────────────────────────────────────────────────────
#  Scene loader  (mirrors JS loadModel() exactly)
# ─────────────────────────────────────────────────────────────────────────────
def load_glb_scene(path):
    print(f"Loading: {path}")
    scene = trimesh.load(path, force="scene")
    if isinstance(scene, trimesh.Trimesh):
        scene = trimesh.Scene({"mesh": scene})

    # ── dump the full scene graph into world-space transformed meshes ──────────
    # This respects node transforms baked in the GLB (fixes sideways models).
    world_meshes = []
    try:
        for node_name in scene.graph.nodes_geometry:
            T, geom_name = scene.graph[node_name]
            geom = scene.geometry.get(geom_name)
            if geom is None or not isinstance(geom, trimesh.Trimesh):
                continue
            if len(geom.faces) == 0:
                continue
            m = geom.copy()
            m.apply_transform(T)
            world_meshes.append((m, geom))
    except Exception:
        # fallback: use geometry dict directly (no transforms)
        for geom in scene.geometry.values():
            if isinstance(geom, trimesh.Trimesh) and len(geom.faces) > 0:
                world_meshes.append((geom, geom))

    if not world_meshes:
        raise ValueError("No renderable meshes found.")

    # ── compute bbox in world space for centering ──────────────────────────────
    all_verts = np.vstack([m.vertices for m, _ in world_meshes])
    bbox_min   = all_verts.min(axis=0)
    bbox_max   = all_verts.max(axis=0)
    center     = ((bbox_min + bbox_max) / 2.0).astype(np.float64)
    model_size = float(np.linalg.norm(bbox_max - bbox_min))
    base_zoom  = model_size * 1.2

    print(f"  BBox diagonal: {model_size:.3f}  |  base zoom dist: {base_zoom:.3f}")

    # ── GLB uses Y-up but some exporters write Z-up — auto-detect & correct ───
    # If height (Y span) < depth (Z span), the model is likely Z-up; rotate -90° X.
    y_span = bbox_max[1] - bbox_min[1]
    z_span = bbox_max[2] - bbox_min[2]
    needs_yup_fix = z_span > y_span * 1.5
    if needs_yup_fix:
        print("  Detected Z-up model — applying Y-up correction (rotate -90° X)")
        # Rotate vertices: new_y = old_z, new_z = -old_y
        fix = np.array([[1, 0, 0],
                        [0, 0, 1],
                        [0,-1, 0]], dtype=np.float64)
    else:
        fix = np.eye(3, dtype=np.float64)

    meshes = []
    for world_mesh, orig_geom in world_meshes:
        verts   = (fix @ (np.array(world_mesh.vertices, dtype=np.float64) - center).T).T
        normals = (fix @ np.array(world_mesh.vertex_normals, dtype=np.float64).T).T
        faces   = np.array(world_mesh.faces, dtype=np.int32)
        try:
            rgba = orig_geom.visual.to_color().vertex_colors.astype(np.float32) / 255.0
            if len(rgba) != len(verts):
                rgba = np.ones((len(verts), 4), np.float32)
        except Exception:
            rgba = np.ones((len(verts), 4), np.float32)
        meshes.append(GLMesh(verts, normals, rgba, faces))

    print(f"  Meshes: {len(meshes)}")
    return meshes, base_zoom


# ─────────────────────────────────────────────────────────────────────────────
#  Main Viewer
# ─────────────────────────────────────────────────────────────────────────────
class ModelViewer(pyglet.window.Window):
    def __init__(self, glb_path):
        super().__init__(
            width=WINDOW_WIDTH, height=WINDOW_HEIGHT,
            caption="GLB Desktop Viewer — Hand Gesture",
            resizable=True,
            config=pyglet.gl.Config(
                double_buffer=True, depth_size=24,
                major_version=3, minor_version=3,
                forward_compatible=True,
            )
        )
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(*BACKGROUND_COLOR)

        self.prog  = build_program()
        self.meshes, self.base_zoom = load_glb_scene(glb_path)
        self._time = 0.0   # shader clock (seconds, drives scanlines & flicker)
        self.stars = StarField()

        # size ring to the model footprint (base_zoom gives a good proxy)
        ring_r = self.base_zoom * 0.38
        self.ring  = GlowRing(
            radius      = ring_r,
            y_offset    = -self.base_zoom * 0.28,   # just below model base
            inner_color = (0.0, 0.7, 1.0),          # cyan centre
            outer_color = (0.5, 0.0, 1.0),          # purple edge
        )

        # particles: spawn in ring around base, rise to model height
        self.particles = ParticleSystem(
            spawn_radius = ring_r * 1.1,
            height       = self.base_zoom * 0.95,
        )

        # ── orbit camera state ────────────────────────────────────────────────
        PHI_DEFAULT       = math.pi / 3    # 60° from top → nice default view
        self.PHI_MIN      = 0.05
        self.PHI_MAX      = math.pi - 0.05

        self.camera_theta = 0.0
        self.camera_phi   = PHI_DEFAULT
        self.target_theta = 0.0
        self.target_phi   = PHI_DEFAULT

        # ── zoom state ────────────────────────────────────────────────────────
        self.current_zoom_scale = 1.0
        self.target_zoom_scale  = 1.0

        # ── mouse fallback state ──────────────────────────────────────────────
        self._mouse_dragging = False

        # ── auto-rotation state ───────────────────────────────────────────────
        self._idle_seconds   = 0.0          # time since last user input
        self._auto_rot_speed = 0.4          # radians per second
        self._auto_rot_blend = 0.0          # 0=off → 1=full (smooth fade-in)
        self._IDLE_THRESHOLD = 3.0          # seconds of no input before spinning

        # ── start gesture thread ──────────────────────────────────────────────
        if HAS_GESTURE:
            threading.Thread(target=run_gesture_thread, daemon=True).start()
            print("[Gesture] Thread started — use pinch gestures to control model")
        else:
            print("[Gesture] Disabled — using mouse controls only")

        pyglet.clock.schedule_interval(self._tick, 1 / 60)
        print(f"\n  Viewer ready: {os.path.basename(glb_path)}")
        print("  Gestures: 1-hand pinch+drag=rotate | 2-hand pinch=zoom")
        print("  Mouse:    LMB drag=rotate | Scroll=zoom | R=reset | Q=quit\n")

    # ── build matrices (model fixed at origin, camera orbits on a sphere) ─────
    def _matrices(self):
        dist = self.base_zoom / self.current_zoom_scale

        # Spherical → Cartesian  (Y-up convention)
        sp, cp = math.sin(self.camera_phi),   math.cos(self.camera_phi)
        st, ct = math.sin(self.camera_theta), math.cos(self.camera_theta)
        eye = np.array([dist * sp * st,
                         dist * cp,
                         dist * sp * ct], dtype=np.float32)

        view  = look_at(eye,
                        np.array([0.0, 0.0, 0.0], dtype=np.float32),
                        np.array([0.0, 1.0, 0.0], dtype=np.float32))
        proj  = perspective(75, self.width / max(self.height, 1),
                            dist * 0.001, dist * 1000)
        model = np.eye(4, dtype=np.float32)
        mvp   = (proj @ view @ model).T.astype(np.float32)
        return mvp, model, eye

    # ── draw ──────────────────────────────────────────────────────────────────
    def on_draw(self):
        self.clear()

        mvp, model, eye = self._matrices()
        # recompute view & proj separately for the star pass
        dist = self.base_zoom / self.current_zoom_scale
        sp, cp = math.sin(self.camera_phi),   math.cos(self.camera_phi)
        st, ct = math.sin(self.camera_theta), math.cos(self.camera_theta)
        eye_v  = np.array([dist*sp*st, dist*cp, dist*sp*ct], dtype=np.float32)
        view   = look_at(eye_v,
                         np.array([0.,0.,0.], dtype=np.float32),
                         np.array([0.,1.,0.], dtype=np.float32))
        proj   = perspective(75, self.width / max(self.height, 1),
                             dist * 0.001, dist * 1000)

        # draw stars first (behind everything, no depth write)
        self.stars.draw(view, proj, self._time)

        # draw ground glow ring (additive, just below model)
        mvp_np = (proj @ view @ np.eye(4, dtype=np.float32)).T.astype(np.float32)
        self.ring.draw(mvp_np, self._time)

        # draw floating particles (additive, rise around model)
        self.particles.draw(view, proj, self._time)

        # draw model
        glUseProgram(self.prog)
        mvp, model, eye = self._matrices()

        def fptr(m):
            return m.flatten().ctypes.data_as(ctypes.POINTER(GLfloat))

        glUniformMatrix4fv(ul(self.prog, "uMVP"),   1, GL_FALSE, fptr(mvp))
        glUniformMatrix4fv(ul(self.prog, "uModel"), 1, GL_FALSE, fptr(model))
        glUniform3f(ul(self.prog, "uLightDir"),   1.0,  2.0,  1.5)
        glUniform3f(ul(self.prog, "uLightDir2"), -1.0, -0.5, -1.0)
        glUniform3f(ul(self.prog, "uCamPos"), *eye)
        glUniform1f(ul(self.prog, "uTime"), self._time)

        for mesh in self.meshes:
            mesh.draw()
        glUseProgram(0)

    # ── tick: lerp camera orbit angles + zoom each frame ─────────────────────
    def _tick(self, dt):
        self._time += dt   # advance shader clock

        # ── idle timer & auto-rotation ────────────────────────────────────────
        # Also treat active gesture input as "not idle"
        if HAS_GESTURE:
            with _gesture_lock:
                g_type = _gesture_state["type"]
                g_time = _gesture_state["timestamp"]
            gesture_active = g_type and (time.time() - g_time) < GESTURE_EXPIRY
        else:
            gesture_active = False

        if self._mouse_dragging or gesture_active:
            self._poke_input()

        self._idle_seconds += dt

        if self._idle_seconds >= self._IDLE_THRESHOLD:
            # Fade blend from 0→1 over 1.5 s so spin-up feels smooth
            self._auto_rot_blend = min(1.0, self._auto_rot_blend + dt / 1.5)
        else:
            self._auto_rot_blend = max(0.0, self._auto_rot_blend - dt / 0.4)

        if self._auto_rot_blend > 0.0:
            self.target_theta += self._auto_rot_speed * dt * self._auto_rot_blend
        # zoom lerp
        self.current_zoom_scale += (self.target_zoom_scale - self.current_zoom_scale) * ZOOM_LERP

        # orbit angle lerp
        self.camera_theta += (self.target_theta - self.camera_theta) * ROT_LERP
        self.camera_phi   += (self.target_phi   - self.camera_phi)   * ROT_LERP
        self.camera_phi    = max(self.PHI_MIN, min(self.PHI_MAX, self.camera_phi))

        # ── apply gesture state ───────────────────────────────────────────────
        if HAS_GESTURE:
            now = time.time()
            with _gesture_lock:
                g = dict(_gesture_state)

            if g["type"] and (now - g["timestamp"]) < GESTURE_EXPIRY:
                if g["type"] == "zoom":
                    self.target_zoom_scale = g["zoom_scale"]

                elif g["type"] == "rotation":
                    # FIXED: negate both axes so model follows hand direction
                    #   rotation_y (from dx) → horizontal → theta  (negate so right=right)
                    #   rotation_x (from dy) → vertical   → phi    (negate so down=down)
                    self.target_theta = -g["rotation_y"] * (math.pi / 180)
                    self.target_phi   = math.pi / 3 - g["rotation_x"] * (math.pi / 180)
                    self.target_phi   = max(self.PHI_MIN, min(self.PHI_MAX, self.target_phi))

    # ── reset idle timer on any user interaction ──────────────────────────────
    def _poke_input(self):
        """Call whenever the user does anything — resets idle timer & fades auto-rot out."""
        self._idle_seconds  = 0.0
        self._auto_rot_blend = 0.0   # snap off so control feels instant

    # ── mouse (fallback / fine control) ───────────────────────────────────────
    def on_mouse_press(self, x, y, button, mods):
        self._poke_input()
        if button == pyglet.window.mouse.LEFT:
            self._mouse_dragging = True

    def on_mouse_release(self, x, y, button, mods):
        if button == pyglet.window.mouse.LEFT:
            self._mouse_dragging = False

    def on_mouse_drag(self, x, y, dx, dy, buttons, mods):
        self._poke_input()
        if self._mouse_dragging and buttons & pyglet.window.mouse.LEFT:
            # FIXED: negate so drag direction matches model movement direction
            self.target_theta -= dx * 0.01          # left/right → horizontal orbit
            self.target_phi   += dy * 0.01          # up/down    → vertical orbit
            self.target_phi    = max(self.PHI_MIN, min(self.PHI_MAX, self.target_phi))

    def on_mouse_scroll(self, x, y, sx, sy):
        self._poke_input()
        delta = 0.1 if sy > 0 else -0.1
        self.target_zoom_scale = max(0.5, min(3.0, self.target_zoom_scale + delta))

    # ── keyboard ──────────────────────────────────────────────────────────────
    def on_key_press(self, sym, mods):
        self._poke_input()
        if sym == pyglet.window.key.R:
            self.camera_theta       = 0.0
            self.camera_phi         = math.pi / 3
            self.target_theta       = 0.0
            self.target_phi         = math.pi / 3
            self.current_zoom_scale = 1.0
            self.target_zoom_scale  = 1.0
            print("[Viewer] Reset")
        elif sym in (pyglet.window.key.Q, pyglet.window.key.ESCAPE):
            self.close()

    def on_resize(self, w, h):
        glViewport(0, 0, w, h)
        return pyglet.event.EVENT_HANDLED

    def on_close(self):
        for m in self.meshes:
            m.delete()
        self.stars.delete()
        self.ring.delete()
        self.particles.delete()
        super().on_close()


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────
def pick_file():
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isfile(path):
            return path
        if os.path.isdir(path):
            files = (glob.glob(os.path.join(path, "**/*.glb"),  recursive=True) +
                     glob.glob(os.path.join(path, "**/*.gltf"), recursive=True))
            if files:
                print(f"Auto-selected: {files[0]}")
                return files[0]

    if HAS_TK:
        root = tk.Tk(); root.withdraw()
        path = filedialog.askopenfilename(
            title="Select a GLB / GLTF file",
            filetypes=[("3D Models", "*.glb *.gltf"), ("All files", "*.*")]
        )
        root.destroy()
        if path:
            return path

    files = (glob.glob("*.glb") + glob.glob("*.gltf") +
             glob.glob("models/*.glb") + glob.glob("models/*.gltf"))
    if files:
        print(f"Auto-detected: {files[0]}")
        return files[0]

    print("Usage:  python glb_desktop_viewer.py  path/to/model.glb")
    sys.exit(1)


if __name__ == "__main__":
    glb_path = pick_file()
    viewer   = ModelViewer(glb_path)
    pyglet.app.run()