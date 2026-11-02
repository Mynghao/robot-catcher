"""MJCF scene: a KUKA LBR iiwa 14, a cup end effector, and a ball.

The arm is generated from Drake's iiwa14 URDF (assets/iiwa14): joint frames, link masses and
inertias, joint position/velocity/torque limits, and the link meshes (visual only).
"""
import os
import xml.etree.ElementTree as ET

import numpy as np
import mujoco

CUP_RADIUS = 0.08   # inner radius
CUP_DEPTH = 0.12
BALL_RADIUS = 0.033
CUP_SOLREF = "0.01 1.5"  # contact time constant, damping ratio: softer/more damped absorbs more
MAX_WALLS = 4
SHOULDER = np.array([0.0, 0.0, 0.36])  # joint-2 location, used for reachability

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "iiwa14")
_URDF = os.path.join(ASSETS, "iiwa14_no_collision.urdf")
FLANGE = 0.045  # media flange: cup mounts this far past link 7 along its z axis


def _rpy_quat(rpy):
    """URDF fixed-axis roll-pitch-yaw -> MuJoCo quaternion (w x y z)."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r / 2), np.sin(r / 2), np.cos(p / 2), np.sin(p / 2), np.cos(y / 2), np.sin(y / 2)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def _parse_urdf():
    """Per-joint dicts (origin, axis, limits, child inertial) for iiwa_joint_1..7, plus the base inertial."""
    root = ET.parse(_URDF).getroot()
    vec = lambda e, a, d="0 0 0": np.array([float(x) for x in (e.get(a, d) if e is not None else d).split()])
    links = {}
    for link in root.findall("link"):
        ine = link.find("inertial")
        if ine is None:
            continue
        o, I = ine.find("origin"), ine.find("inertia")
        links[link.get("name")] = dict(
            com=vec(o, "xyz"), mass=float(ine.find("mass").get("value")),
            I=[float(I.get(k)) for k in ("ixx", "iyy", "izz")])
    joints = []
    for i in range(1, 8):
        j = root.find(f"joint[@name='iiwa_joint_{i}']")
        o, lim = j.find("origin"), j.find("limit")
        joints.append(dict(
            xyz=vec(o, "xyz"), quat=_rpy_quat(vec(o, "rpy")), axis=vec(j.find("axis"), "xyz", "0 0 1"),
            lower=float(lim.get("lower")), upper=float(lim.get("upper")),
            velocity=float(lim.get("velocity")), effort=float(lim.get("effort")),
            inertial=links[j.find("child").get("link")]))
    return links["iiwa_link_0"], joints


_BASE, _JOINTS = _parse_urdf()
QD_LIMITS = np.array([j["velocity"] for j in _JOINTS])  # rad/s, per joint
TAU_LIMITS = np.array([j["effort"] for j in _JOINTS])   # N m, per joint


def _cup_xml(n_panels=16):
    # Cup frame: origin at the bottom, +z is the opening direction.
    wall = 0.006
    h = CUP_DEPTH / 2
    parts = [f'<geom class="cup" type="cylinder" size="{CUP_RADIUS + wall} {wall/2}" pos="0 0 {wall/2}"/>']
    half_w = (CUP_RADIUS + wall) * np.tan(np.pi / n_panels) + 0.002
    for i in range(n_panels):
        a = 2 * np.pi * i / n_panels
        r = CUP_RADIUS + wall / 2
        parts.append(
            f'<geom class="cup" type="box" size="{wall/2} {half_w:.4f} {h}" '
            f'pos="{r*np.cos(a):.4f} {r*np.sin(a):.4f} {h + wall}" euler="0 0 {a:.4f}"/>'
        )
    parts.append(f'<site name="cup_mouth" pos="0 0 {CUP_DEPTH + wall}" size="0.01"/>')
    parts.append(f'<site name="cup_bottom" pos="0 0 {wall}" size="0.01"/>')
    return "\n".join(parts)


def build_xml():
    # Walls are mocap bodies placed per episode. They're visual only: stage2.py computes the
    # ball's wall bounces itself, so restitution is exact and per wall.
    walls = "\n".join(
        f'<body name="wall{i}" mocap="true" pos="0 0 -5"><geom name="wall{i}" type="box" '
        f'size="1 1 0.01" contype="0" conaffinity="0" rgba="0.5 0.7 0.6 0.45"/></body>'
        for i in range(MAX_WALLS)
    )
    fmt = lambda a: " ".join(f"{x:.9g}" for x in a)
    inertial = lambda d: (f'<inertial pos="{fmt(d["com"])}" mass="{d["mass"]}" '
                          f'diaginertia="{fmt(d["I"])}"/>')
    assets = "\n".join(f'<mesh name="link_{i}" file="link_{i}.obj"/>' for i in range(8))
    arm = ""
    close = ""
    for i, j in enumerate(_JOINTS):
        arm += (
            f'<body name="link{i+1}" pos="{fmt(j["xyz"])}" quat="{fmt(j["quat"])}" gravcomp="1">\n'
            f'{inertial(j["inertial"])}\n'
            f'<joint name="j{i+1}" axis="{fmt(j["axis"])}" '
            f'range="{np.rad2deg(j["lower"]):.4f} {np.rad2deg(j["upper"]):.4f}"/>\n'
            f'<geom class="link" mesh="link_{i+1}"/>\n'
        )
        close += "</body>\n"
    ee = f'<body name="cup" pos="0 0 {FLANGE}" gravcomp="1">{_cup_xml()}</body>\n'
    actuators = "\n".join(
        f'<position joint="j{i+1}" kp="3000" kv="120" ctrlrange="{j["lower"]:.4f} {j["upper"]:.4f}" '
        f'forcerange="{-j["effort"]} {j["effort"]}"/>'
        for i, j in enumerate(_JOINTS)
    )
    return f"""
<mujoco model="wall_catching_iiwa14">
  <compiler angle="degree" meshdir="{ASSETS}"/>
  <asset>{assets}</asset>
  <option timestep="0.002" integrator="implicitfast"/>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <default>
    <joint damping="2" armature="0.1"/>
    <default class="link"><geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.95 0.45 0.1 1"/></default>
    <default class="cup"><geom contype="1" conaffinity="1" rgba="0.2 0.4 0.9 0.6" mass="0.03"
        solref="{CUP_SOLREF}" solimp="0.95 0.99 0.001" friction="1.0 0.02 0.001"/></default>
  </default>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="6 6 0.1" rgba="0.8 0.8 0.8 1"/>
    <body name="base">{inertial(_BASE)}<geom class="link" mesh="link_0" rgba="0.4 0.4 0.4 1"/>
    {arm}{ee}{close}
    </body>
    {walls}
    <body name="ball" pos="3 0 1">
      <freejoint name="ball"/>
      <geom name="ball" type="sphere" size="{BALL_RADIUS}" mass="0.057" rgba="0.9 0.9 0.1 1"
            solref="0.01 1.5" friction="1.0 0.02 0.001"/>
    </body>
  </worldbody>
  <actuator>{actuators}</actuator>
</mujoco>
"""


def load_model():
    return mujoco.MjModel.from_xml_string(build_xml())
