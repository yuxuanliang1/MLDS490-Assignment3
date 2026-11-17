"""One-time correction of the identified duplicate driver; never touches other jobs."""
import psutil
p=psutil.Process(844766)
assert 'part1_2.workflow' in p.cmdline() and 'validate' in p.cmdline()
children=p.children(recursive=True)
for c in children:
    try:c.kill()
    except psutil.NoSuchProcess:pass
p.kill()
psutil.wait_procs(children,timeout=5)
print('Stopped only duplicate driver 844766 and its descendants')
