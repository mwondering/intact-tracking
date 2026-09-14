"""Verify the new eight-DR bank using the established batch/reset audit."""

import verify_fixed_dr_profiles as verify
from intact_tracking.residual_uniform_protocol import configure_physics, audit_physics


if __name__ == "__main__":
    verify.configure_fixed_dr = configure_physics
    verify.audit_fixed_dr = audit_physics
    verify.main()
