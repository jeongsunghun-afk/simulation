"""17-DOF(260701) URDF → MuJoCo MJCF 빌드. build_real_quad.py 파생.
   차이: ①소스=새 패키지(02_Leg_UFDF_260701) ②base 링크명 "Base" ③FB_waist_joint→fixed(잠금)
        ④기존 모델 안 건드리게 별도 출력(quad_real_17dof.mjcf, meshes_sim_17dof/)
   결과: 16-DOF 전(全)발목 4족 + 허리 고정. (sphere발 변형은 후처리 별도)
"""
import os, re, sys, glob
import numpy as np
import xml.etree.ElementTree as ET
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
PKG  = '/home/jsh/문서/jsh/simulation/02_Leg_UFDF_260701'
SRC_MESH = os.path.join(PKG, 'meshes')
SRC_URDF = os.path.join(PKG, 'urdf', '02_Leg_UFDF_260701_3.urdf')
MESH_OUT = os.path.join(HERE, 'meshes_sim_17dof')
MJCF_OUT = os.path.join(HERE, 'quad_real_17dof.mjcf')
TARGET = 60000
FORCE = '--force' in sys.argv
print('URDF:', os.path.basename(SRC_URDF))

# ── ① decimation (새 메시 → meshes_sim_17dof) ──
os.makedirs(MESH_OUT, exist_ok=True)
n_done = 0
for f in sorted(glob.glob(SRC_MESH + '/*.STL')):
    out = os.path.join(MESH_OUT, os.path.basename(f))
    if (not FORCE) and os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(f):
        continue
    try:
        import trimesh
        mesh = trimesh.load(f, force='mesh'); n0 = len(mesh.faces)
        if n0 > TARGET:
            try: mesh = mesh.simplify_quadric_decimation(face_count=TARGET)
            except TypeError: mesh = mesh.simplify_quadric_decimation(TARGET)
        mesh.export(out); print('  %-32s %7d → %6d' % (os.path.basename(f), n0, len(mesh.faces)))
    except ModuleNotFoundError:
        import shutil; shutil.copy(f, out); print('  %-32s (그대로 복사, trimesh 없음)' % os.path.basename(f))
    n_done += 1
print('decimation: %d개 갱신' % n_done if n_done else 'decimation: 전부 최신 → 건너뜀')

# ── ② URDF 전처리: package 제거 + compiler + ★FB_waist_joint→fixed ──
u = open(SRC_URDF).read()
u = re.sub(r'package://[^/]+/meshes/', '', u)
# 허리 잠금: FB_waist_joint 블록의 type="revolute"→"fixed" (해당 joint만)
def _lock_waist(txt):
    # <joint ... name="FB_waist_joint" ... type="revolute" ...> — 멀티라인, name/type 순서 무관
    def repl(m):
        blk = m.group(0)
        if 'FB_waist_joint' in blk:
            blk = blk.replace('type="revolute"', 'type="fixed"')
        return blk
    return re.sub(r'<joint\b.*?</joint>', repl, txt, flags=re.S)
u = _lock_waist(u)
tag = (f'<mujoco><compiler meshdir="{MESH_OUT}" balanceinertia="true" '
       f'discardvisual="false" fusestatic="false"/></mujoco>')
u = re.sub(r'(<robot[^>]*>)', r'\1\n  ' + tag, u, count=1)
open('/tmp/quad17_build.urdf', 'w').write(u)

# ── ③ URDF→MJCF + freejoint + 지면 + actuator(모든 관절 제네릭) ──
m0 = mujoco.MjModel.from_xml_path('/tmp/quad17_build.urdf')
mujoco.mj_saveLastXML('/tmp/quad17_build.mjcf', m0)
tree = ET.parse('/tmp/quad17_build.mjcf'); root = tree.getroot()
wb = root.find('worldbody')
base = wb.find('body')          # ★root body(이름 Base/base 무관, 첫 body)
base.insert(0, ET.Element('freejoint', {'name': 'root'}))
base.set('pos', '0 0 0.6')
ET.SubElement(wb, 'geom', {'name': 'floor', 'type': 'plane', 'size': '5 5 0.1', 'rgba': '0.4 0.5 0.4 1'})
ET.SubElement(wb, 'light', {'pos': '0 0 3', 'dir': '0 0 -1'})
ET.SubElement(root, 'option', {'timestep': '0.002', 'gravity': '0 0 -9.81'})
dft = ET.SubElement(root, 'default')
ET.SubElement(dft, 'geom', {'friction': '1.3 0.02 0.001', 'condim': '3'})
ET.SubElement(dft, 'motor', {'ctrllimited': 'true', 'ctrlrange': '-80 80'})
act = ET.SubElement(root, 'actuator')
for jn in [j.get('name') for j in root.iter('joint') if j.get('name')]:
    ET.SubElement(act, 'motor', {'joint': jn, 'name': jn.replace('_joint', '')})
ET.indent(tree, space='  '); tree.write(MJCF_OUT, encoding='unicode')

# ── ④ 충돌 최저점 → 기립높이 보정 ──
m = mujoco.MjModel.from_xml_path(MJCF_OUT); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
zmin = 1e9
for g in range(m.ngeom):
    if (m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0) or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
        continue
    mid = m.geom_dataid[g]; adr = m.mesh_vertadr[mid]; nv = m.mesh_vertnum[mid]
    V = m.mesh_vert[adr:adr + nv].reshape(-1, 3); R = d.geom_xmat[g].reshape(3, 3)
    zmin = min(zmin, float(((V @ R.T)[:, 2] + d.geom_xpos[g][2]).min()))
need = 0.6 - zmin + 0.002
base.set('pos', '0 0 %.4f' % need); tree.write(MJCF_OUT, encoding='unicode')
print('충돌 최저 z=%.3f → 기립 base_z=%.4f' % (zmin, need))
print('%s 완료: nq=%d nv=%d nu=%d 총질량=%.1fkg' % (os.path.basename(MJCF_OUT), m.nq, m.nv, m.nu, m.body_subtreemass[0]))
# 관절 순서(제어기 브리지용)
_jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt) if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
print('관절순서(%d):' % len(_jn), _jn)
