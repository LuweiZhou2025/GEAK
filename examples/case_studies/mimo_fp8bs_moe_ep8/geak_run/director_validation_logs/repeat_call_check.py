# Director-side extra check (not part of the frozen oracle): repeated alternating calls, like the perf loop,
# then verify outputs of later calls against the frozen reference with the frozen tolerances.
import importlib.util, sys, os
ws = sys.argv[1]
spec = importlib.util.spec_from_file_location("task_runner_v", os.path.join(ws, "scripts", "task_runner.py"))
tr = importlib.util.module_from_spec(spec); spec.loader.exec_module(tr)
import torch
impl = tr.load_impl()
raw = tr.make_weights()
prepared = impl.prepare_weights(*(t.clone() for t in raw))
ok_all = True
for i, (cid, tpr, max_disp) in enumerate(tr.CASES):
    sets = [tr.make_inputs(tpr, max_disp, seed=s) for s in (100 + i, 200 + i)]
    refs = [tr.reference(*raw, inp, n) for inp, n in sets]
    for k in range(7):  # outputs discarded -> allocator recycles dirty blocks
        impl.forward(prepared, **sets[k % 2][0])
    torch.cuda.synchronize()
    for j in (0, 1, 0):
        out = impl.forward(prepared, **sets[j][0]); torch.cuda.synchronize()
        ok, msg = tr.compare(out, refs[j], sets[j][1])
        print(f"{'PASS' if ok else 'FAIL'} {cid} set{j} (after repeated calls): {msg}", flush=True)
        ok_all = ok_all and ok
        del out
print(f"RepeatCall: {'PASS' if ok_all else 'FAIL'}")
sys.exit(0 if ok_all else 1)
