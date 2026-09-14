# Incomplete engineering attempt — not eligible for the ABC result

A stopped advancing after 20 updates and B after 33. The first run did not
have an exception handler that printed a failing rank's exception before
destroying its NCCL process group. A subsequent B diagnostic run reproduced
one rank stuck in `destroy_process_group` while the peer waited during rollout.

All three jobs in this directory were stopped early by the experiment runner.
No 1000-update arm or final endpoint comparison was completed in this directory.
W&B runs A/B/C were marked `aborted_multigpu_stall` and
`eligible_endpoint_result=False`. Files are retained for diagnosis.

The corrected launcher prints the original exception and exits the failed rank
immediately, so torchrun can stop its peer instead of silently waiting for NCCL.
This is an error-reporting fix, not a change to the reward or learning rule.
