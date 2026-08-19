# BMD Agent Security Model

## 1. Core rule

BMD Agent v0 has no authority to mutate authoritative BMD resources.

This is a system design requirement, not merely an instruction to the
language model.

Observation and action are separate security domains.

## 2. Governance

BMD Agent is PI-governed.

Students and other group members may:

- use the agent;
- inspect results;
- conduct investigations;
- propose improvements;
- contribute code to the BMD Agent project;
- use agent development as a learning exercise.

They may not use the agent to make authoritative changes without PI
approval.

Future action capabilities must preserve this boundary.

## 3. Protected resources

Protected resources include, but are not limited to:

- the live BMD Compute checkout;
- BMDex;
- BMDwiki;
- student and researcher project files;
- PowerSLURM jobs;
- BMD scientific datasets;
- deployment configuration;
- Git remotes;
- public-facing BMD resources.

## 4. BMD Compute

The local BMD Compute checkout is a live production resource.

Current location:

    ~/projects/bmd_compute

The running BMD Compute service uses this checkout.

The agent may inspect it but must not modify it in v0.

Examples of allowed operations:

- read source;
- read configuration;
- inspect Git status;
- inspect Git history;
- inspect commit identity;
- inspect tests;
- search files.

Examples of prohibited operations:

- edit files;
- git pull;
- git checkout;
- git reset;
- git clean;
- git commit;
- git push;
- modify the Python environment;
- restart Uvicorn;
- deploy changes.

A dirty working tree must be reported, not "fixed."

## 5. BMDex

BMDex is authoritative for curated BMD scientific knowledge and data.

BMD Agent v0 may read and search BMDex.

It may not:

- modify BMDex records;
- promote an observation into a standard;
- commit;
- push;
- merge;
- declare a standard adopted.

Scientific observations made by the agent do not automatically become
BMD knowledge.

## 6. PowerSLURM

BMD Agent connects to PowerSLURM using the dedicated guest identity.

SSH alias:

    powerslurm-bmdguest

Remote identity:

    bmdguest

Normal scheduler inspection should be restricted to the BMD group
scope:

    leeburton-pool

The agent should not expose unrelated TAU cluster activity merely
because SLURM permits broad scheduler visibility.

Allowed cluster operations include:

- inspect queue state;
- inspect job metadata;
- inspect accounting information;
- read authorized calculation files;
- inspect VASP inputs and outputs;
- inspect scheduler logs;
- copy information into an explicitly designated agent workspace when
  necessary for analysis.

Prohibited operations include:

- sbatch;
- scancel;
- scontrol update or other state-changing scheduler operations;
- modifying researcher files;
- deleting researcher files;
- moving or renaming researcher files;
- changing permissions;
- altering running calculations.

## 7. Agent workspace

The agent may require writable storage for:

- temporary copies;
- parsed data;
- derived analyses;
- reports;
- indexes;
- task state;
- logs.

Writable agent storage must be explicitly designated.

Writable agent storage is not authoritative BMD scientific storage.

The agent must not treat an artifact as validated merely because it
exists in its workspace.

## 8. Command execution

The long-term public tool interface should prefer explicit operations
over unrestricted shell access.

Prefer interfaces such as:

    get_bmd_queue()
    get_job(job_id)
    inspect_repository(name)
    read_calculation(path)
    parse_vasp_output(path)

rather than:

    shell(command)

Internal development may temporarily use lower-level mechanisms, but the
security model must not depend on the language model voluntarily
avoiding dangerous commands.

## 9. Least privilege

Where practical, security should be enforced by:

- Unix users and groups;
- filesystem permissions;
- ACLs;
- restricted SSH identities;
- Git/GitHub permissions;
- explicit tool allowlists;
- application authorization.

Prompt instructions are not a security boundary.

The persistent production agent should eventually run under a dedicated
VM service identity rather than inheriting the PI's personal
credentials and filesystem authority.

## 10. Credentials

The agent must not unnecessarily inherit:

- personal GitHub credentials;
- personal SSH credentials;
- cluster write credentials;
- deployment credentials.

Credentials should be scoped to the minimum capabilities required.

Read credentials and write credentials should be separate where
possible.

## 11. Future action plane

Action capabilities may be introduced later.

Examples include:

- preparing changes in an isolated worktree;
- committing;
- pushing;
- opening pull requests;
- submitting validation calculations;
- cancelling jobs;
- deploying services;
- publishing documentation.

Introducing such a capability requires:

1. an explicit tool implementation;
2. a defined authorization policy;
3. identification of the requesting user;
4. an explicit approval state;
5. PI authorization where required;
6. an audit record;
7. clear reporting of the resulting action.

The ability of the underlying operating-system account to perform an
operation does not imply that BMD Agent is authorized to perform it.

## 12. Auditability

Material agent operations should eventually record:

- requester;
- task;
- timestamp;
- resource inspected;
- operation performed;
- inputs;
- artifacts produced;
- evidence used;
- failures;
- conclusions;
- recommendations;
- approvals;
- resulting Git commits or computational jobs where applicable.

The objective is to make it possible to reconstruct:

- what the agent inspected;
- what it concluded;
- why it reached that conclusion;
- what it changed, if anything;
- what tests it ran;
- what scientific evidence supported its claims.

## 13. Scientific safety

The agent must distinguish observation from inference.

It should preserve important methodological distinctions such as:

    ML prediction != first-principles result
    first-principles result != experiment
    calculation completed != calculation validated
    workflow implemented != workflow validated
    database absence != proven novelty
    negative formation energy != convex-hull stability

When evidence is incomplete, the agent should state the validation gap
rather than silently promote the claim.

## 14. v0 guarantee

The simplest security property of v0 is:

    no mutation tools exist

The first implementation should prove that useful scientific
observation, troubleshooting, and advice are possible before any action
capabilities are introduced.
