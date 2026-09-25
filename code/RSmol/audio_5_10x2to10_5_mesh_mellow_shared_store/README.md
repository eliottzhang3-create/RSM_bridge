# Variable-depth Audio MeSH shared-store route

This package is isolated from the fixed two-loop implementation. It preserves
the audited ReasonAQA fixed-260-token two-slot audio pipeline and the twenty
physical 5-10-5 decoder modules, while one integer recursive depth is sampled
uniformly from 2 through 10 for each micro-step and broadcast from rank zero.

Router migration from the fixed checkpoint is explicit: pre routers come from
the old pre routers, loop1 routers from old loop 0, refine_write and out_read
from old loop 1, and the only new router (refine_read) is initialized from old
loop-0 read. At depth two, refine_read is intentionally unused.

The eventual formal schedule defaults to 7 epochs with warmup equal to 5% of
the real optimizer-step count (rounded up). Phase 1--4 utilities in
scripts create the initialization artifact, audit every structural depth
automatically, and run a real depth-10 activation-memory preflight before any
training smoke/resume/formal job is enabled.
