from audit_env.policies.flag_all import FlagAllPolicy
from audit_env.policies.oracle import OraclePolicy
from audit_env.policies.random_explorer import RandomExplorerPolicy
from audit_env.policies.reconciler import ReconcilerPolicy, RulesAuditorPolicy

POLICIES = {p.name: p for p in (RandomExplorerPolicy, ReconcilerPolicy, FlagAllPolicy, OraclePolicy,
                                RulesAuditorPolicy)}
BASELINES = ("random", "reconciler", "flag_all", "oracle")

__all__ = ["BASELINES", "POLICIES", "RulesAuditorPolicy", "FlagAllPolicy", "OraclePolicy", "RandomExplorerPolicy", "ReconcilerPolicy"]
