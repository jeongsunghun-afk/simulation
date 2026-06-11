"""실제 4족 URDF → MuJoCo MJCF 통합 빌드.  (자체 완결형 프로젝트: simulation/quad/)
   소스 URDF=urdf/, 원본 메시=meshes/ (이 폴더), 결과 quad_real.mjcf + meshes_sim/ 생성.

   ① 고해상 STL decimation → meshes_sim/  (이미 있으면 건너뜀)
   ② package:// 경로 해제 + <mujoco> compiler 태그
   ③ URDF→MJCF, base에 freejoint + 지면
   ④ foot_contact_link 메시(충돌 박스)로 발 접지 — 메시 최저점으로 기립높이 보정
   ⑤ 16관절 motor 액추에이터 + 발 마찰
"""
import os
import re
import sys
import glob
import numpy as np
import xml.etree.ElementTree as ET
import trimesh
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_MESH = os.path.join(HERE, 'meshes')
MESH_OUT = os.path.join(HERE, 'meshes_sim')
MJCF_OUT = os.path.join(HERE, 'quad_real.mjcf')
TARGET = 60000          # decimation 목표 face (MuJoCo 한계 20만 이하, 시각 품질↑)
FORCE = '--force' in sys.argv

# URDF 자동 탐지 (urdf/ 안의 .urdf 1개) — 파일명 바뀌어도 동작
_urdfs = sorted(glob.glob(os.path.join(HERE, 'urdf', '*.urdf')))
assert _urdfs, 'urdf/ 폴더에 .urdf 파일이 없습니다'
SRC_URDF = _urdfs[0]
print('URDF:', os.path.basename(SRC_URDF))

# ── ① decimation (소스 메시가 더 최신인 것만 재처리; --force 면 전부) ──
os.makedirs(MESH_OUT, exist_ok=True)
n_done = 0
for f in sorted(glob.glob(SRC_MESH + '/*.STL')):
    out = os.path.join(MESH_OUT, os.path.basename(f))
    if (not FORCE) and os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(f):
        continue   # 최신이면 skip
    mesh = trimesh.load(f, force='mesh')
    n0 = len(mesh.faces)
    if n0 > TARGET:
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=TARGET)
        except TypeError:
            mesh = mesh.simplify_quadric_decimation(TARGET)
    mesh.export(out)
    print('  %-32s %7d → %6d' % (os.path.basename(f), n0, len(mesh.faces)))
    n_done += 1
print('decimation: %d개 갱신 (나머지 최신)' % n_done if n_done else 'decimation: 전부 최신 → 건너뜀')

# ── ② URDF 전처리 ─────────────────────────────────────
u = open(SRC_URDF).read()
u = re.sub(r'package://[^/]+/meshes/', '', u)   # 어떤 package 이름이든 제거
tag = (f'<mujoco><compiler meshdir="{MESH_OUT}" balanceinertia="true" '
       f'discardvisual="false" fusestatic="false"/></mujoco>')
u = re.sub(r'(<robot[^>]*>)', r'\1\n  ' + tag, u, count=1)
open('/tmp/quad_build.urdf', 'w').write(u)

# ── ③ URDF→MJCF + freejoint + 지면 ────────────────────
m0 = mujoco.MjModel.from_xml_path('/tmp/quad_build.urdf')
mujoco.mj_saveLastXML('/tmp/quad_build.mjcf', m0)
tree = ET.parse('/tmp/quad_build.mjcf'); root = tree.getroot()
wb = root.find('worldbody'); base = wb.find("body[@name='base']")
base.insert(0, ET.Element('freejoint', {'name': 'root'}))
base.set('pos', '0 0 0.6')
ET.SubElement(wb, 'geom', {'name': 'floor', 'type': 'plane', 'size': '5 5 0.1',
                           'rgba': '0.4 0.5 0.4 1'})
ET.SubElement(wb, 'light', {'pos': '0 0 3', 'dir': '0 0 -1'})
ET.SubElement(root, 'option', {'timestep': '0.002', 'gravity': '0 0 -9.81'})

dft = ET.SubElement(root, 'default')
ET.SubElement(dft, 'geom', {'friction': '0.9 0.02 0.001', 'condim': '3'})
ET.SubElement(dft, 'motor', {'ctrllimited': 'true', 'ctrlrange': '-80 80'})
act = ET.SubElement(root, 'actuator')
for jn in [j.get('name') for j in root.iter('joint') if j.get('name')]:
    ET.SubElement(act, 'motor', {'joint': jn, 'name': jn.replace('_joint', '')})

ET.indent(tree, space='  '); tree.write(MJCF_OUT, encoding='unicode')

# ── ④ 충돌 geom 최저점으로 기립높이 보정 ──────────────
m = mujoco.MjModel.from_xml_path(MJCF_OUT); d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
zmin = 1e9
for g in range(m.ngeom):
    if (m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0) \
            or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
        continue
    mid = m.geom_dataid[g]; adr = m.mesh_vertadr[mid]; nv = m.mesh_vertnum[mid]
    V = m.mesh_vert[adr:adr + nv].reshape(-1, 3)
    R = d.geom_xmat[g].reshape(3, 3)
    zmin = min(zmin, float(((V @ R.T)[:, 2] + d.geom_xpos[g][2]).min()))
need = 0.6 - zmin + 0.002
base.set('pos', '0 0 %.4f' % need)
tree.write(MJCF_OUT, encoding='unicode')
print('충돌 최저 z=%.3f → 기립 base_z=%.4f' % (zmin, need))
print('%s 완료: nq=%d nv=%d nu=%d 총질량=%.1fkg' %
      (os.path.basename(MJCF_OUT), m.nq, m.nv, m.nu, m.body_subtreemass[0]))
