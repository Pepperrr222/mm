"""Independent causal and physical contract checks for the rolling experiment."""
import copy
import numpy as np
import pandas as pd
import run as r

def main():
    data,a=r.prepare()
    # Poison all unavailable daily observations, preserving available history.
    original=r.q2.CausalForecaster(data)
    masked=copy.deepcopy(data)
    masked.load_kw[31:]=np.nan
    masked.pv_kw[31:]=np.nan
    poisoned=r.q2.CausalForecaster(masked)
    clean=original._forecast_signal('load',31)
    dirty=poisoned._forecast_signal('load',31)
    for i in (0,1):
        assert np.array_equal(clean[i],dirty[i])
        assert np.isfinite(dirty[i]).all()
    # Integration of a linear ramp; right-endpoint targets must not shift it.
    linear,pchip=r.pv_interpolate(0,np.arange(1,25,dtype=float))
    expected=(np.arange(144)+.5)/6
    np.testing.assert_allclose(linear,expected,atol=1e-12)
    np.testing.assert_allclose(pchip,expected,atol=1e-12)
    # Safety projection with positive, negative and capacity-limited net demand.
    r.q2.run_execution_unit_tests()
    for soc in (1200.,6000.,10800.):
        for net in (-2000.,0.,2000.):
            c,d,h,w,e=r.q2.execute_interval(soc,300.,r.Q,r.Q,max(net,0),max(-net,0))
            assert abs(300+d+h-net-c-w)<1e-6
            assert 1200-1e-6<=e<=10800+1e-6
            assert not(c>1e-6 and d>1e-6)
    # Settlement: 100 -> 120 -> 105, charged once relative to each prior order.
    ledger=[(100,120),(120,105)]
    increment=sum(1.5*max(new-old,0)-.5*max(old-new,0) for old,new in ledger)
    assert increment==22.5
    gross=sum(abs(new-old) for old,new in ledger)
    assert gross-abs(105-100)==30
    # Every chosen cross-midnight residual path is complete by the current issue.
    for day in range(31,365):
        for ix,k in enumerate((0,36,72,108)):
            chosen=a['chosen'][day,ix]
            assert len(set(chosen))==10 and (chosen<day).all()
            assert np.all(chosen*144+k+144<=day*144+k)
            assert np.isfinite(a['residual_linear_0'][chosen,ix]).all()
    # Independent original-input hashes, stored before experimental writes.
    sources=[r.q2.INPUT1,r.q2.INPUT2,r.ROOT/'01_题目与数据/原始附件/附件3.xlsx',r.ROOT/'01_题目与数据/原始附件/附件5/result3.xlsx']
    r.dump({str(p.relative_to(r.ROOT)):r.sha(p) for p in sources},r.OUT/'protected_inputs_hashes.json')
    r.dump({'future_poison_test':'pass','interpolation_ramp':'pass','projection_boundaries':'pass','previous_commitment_settlement':'pass','complete_historical_paths':'pass'},r.OUT/'contract_tests.json')
    print('Contract tests passed.',flush=True)

if __name__=='__main__':main()
