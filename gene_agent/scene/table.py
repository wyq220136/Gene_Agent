"""水平桌面配置：米制中心 XY、顶高、长宽和厚度，供编辑与各后端共用。"""
import numpy as np


def validate_table(value, defaults=None):
    if not isinstance(value, dict):
        raise ValueError('桌面配置必须是对象')
    table = {**(defaults or {}), **value}
    result = {}
    for key in ('center_xy_m', 'size_xy_m'):
        v = np.asarray(table.get(key), dtype=float)
        if v.shape != (2,) or not np.isfinite(v).all():
            raise ValueError(f'桌面 {key} 需要两个有限米制数值')
        if key == 'size_xy_m' and (np.any(v < .05) or np.any(v > 5)):
            raise ValueError('桌面长宽范围为 0.05–5 m')
        if key == 'center_xy_m' and np.any(np.abs(v) > 3):
            raise ValueError('桌面中心范围为 ±3 m')
        result[key] = v.tolist()
    for key, default in (('top_z_m', None), ('thickness_m', .04)):
        try:
            v = float(table.get(key, default))
        except (TypeError, ValueError):
            raise ValueError(f'桌面 {key} 需要有限米制数值') from None
        if not np.isfinite(v) or (key == 'top_z_m' and abs(v) > 3) or (
                key == 'thickness_m' and not .005 <= v <= .5):
            raise ValueError('桌面顶高范围 ±3 m，厚度范围 0.005–0.5 m')
        result[key] = v
    return result


def table_bounds_xy(table):
    x, y = table['center_xy_m']; sx, sy = table['size_xy_m']
    return [x-sx/2, x+sx/2, y-sy/2, y+sy/2]
