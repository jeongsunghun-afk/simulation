"""quad_real_17dof.mjcf → sphere발 변형(quad_real_17dof_sphere.mjcf).
   각 foot_contact_link: 충돌메시를 visual-only(contype=0)로 + {L}_sphere를 ★프레임 원점(0 0 0)에 추가.
   (새 모델은 foot_contact_link 좌표계가 이미 끝점=접촉점 → 오프셋 불필요. 구 모델은 70mm 위라 오프셋 뒀음)
"""
import os, numpy as np, xml.etree.ElementTree as ET, mujoco
HERE = os.path.dirname(os.path.abspath(__file__))
MJCF = os.path.join(HERE, 'quad_real_17dof.mjcf')
OUT  = os.path.join(HERE, 'quad_real_17dof_sphere.mjcf')
LEGS = ['HL', 'HR', 'FL', 'FR']
RAD = 0.018

# XML 편집: 충돌메시 → visual-only, sphere를 프레임 원점에 추가
tree = ET.parse(MJCF); root = tree.getroot()
base = root.find('worldbody').find('body')
for body in root.iter('body'):
    nm = body.get('name')
    if nm and nm.endswith('_foot_contact_link'):
        L = nm.split('_')[0]
        for geom in body.findall('geom'):
            if geom.get('type') == 'mesh' and geom.get('contype') is None:   # 충돌메시 → visual-only
                geom.set('contype', '0'); geom.set('conaffinity', '0')
        ET.SubElement(body, 'geom', {'name': f'{L}_sphere', 'type': 'sphere', 'size': str(RAD),
            'pos': '0 0 0', 'rgba': '0.9 0.3 0.3 1',           # ★끝점 좌표계 원점에 바로
            'friction': '1.3 0.02 0.001', 'condim': '3'})
ET.indent(tree, space='  '); tree.write(OUT, encoding='unicode')

# 기립높이 보정(sphere 최저점 → 지면)
m2 = mujoco.MjModel.from_xml_path(OUT); d2 = mujoco.MjData(m2); mujoco.mj_forward(m2, d2)
zmin = 1e9
for g in range(m2.ngeom):
    if m2.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE and m2.geom_contype[g] != 0:
        zmin = min(zmin, float(d2.geom_xpos[g][2] - m2.geom_size[g][0]))
cur = float(base.get('pos').split()[2]); need = cur - zmin + 0.002
base.set('pos', '0 0 %.4f' % need); tree.write(OUT, encoding='unicode')
m3 = mujoco.MjModel.from_xml_path(OUT)
print('sphere 최저 z=%.3f → 기립 base_z=%.4f' % (zmin, need))
print('%s 완료: nq=%d nv=%d nu=%d' % (os.path.basename(OUT), m3.nq, m3.nv, m3.nu))
