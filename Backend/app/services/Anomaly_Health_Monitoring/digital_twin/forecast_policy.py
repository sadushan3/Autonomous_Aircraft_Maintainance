from __future__ import annotations
import math

DEFAULT_HEALTHY_GUARD={"max_r2_drop":.005,"max_rmse_ratio":1.01,"max_mae_ratio":1.01}

def healthy_forecast_guard(before,after,policy=None):
    limits=DEFAULT_HEALTHY_GUARD if policy is None else policy
    if not all(math.isfinite(float(value[k])) for value in (before,after) for k in ('r2','rmse','mae')):
        return False
    return (after['r2']>=before['r2']-limits['max_r2_drop']
        and after['rmse']<=before['rmse']*limits['max_rmse_ratio']
        and after['mae']<=before['mae']*limits['max_mae_ratio'])
