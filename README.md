# BMD Agent

BMD Agent is a persistent, physics-grounded scientific reference,
troubleshooting, and coordination agent for the BMD Lab at Tel Aviv
University.

It is intended to help researchers understand and use the group's
computational infrastructure, scientific knowledge, research data,
documentation, and software while preserving clear boundaries between
those systems.

BMD Agent is an independent project. It is not part of BMD Compute,
BMDex, BMDwiki, or the BMD Lab website.

## BMD ecosystem

The BMD software and knowledge ecosystem has distinct components with
different responsibilities:

- **BMD Compute — generate**
  - Browser-based computational interface.
  - Generates new computational materials data.
  - Uses pymatgen, atomate2/jobflow, VASP, and the TAU PowerSLURM
    cluster.

- **BMDex — preserve**
  - Curated scientific knowledge and datasets that the group wants to
    carry forward.
  - Records computational standards, methodology, provenance,
    validation, evidence, and limitations.

- **BMDwiki — explain**
  - Human-oriented documentation.
  - Tutorials, explanations, troubleshooting guides, onboarding, and
    educational material.

- **BMD Lab website — present and connect**
  - Public face of the group.
  - Presents people, publications, teaching, news, and links to BMD
    resources.

- **BMD Agent — understand, diagnose, advise, and coordinate**
  - Reasons across the BMD ecosystem.
  - Helps researchers find information, understand calculations,
    troubleshoot problems, identify evidence, and determine appropriate
    next steps.

These systems remain independently version-controlled and authoritative
for their respective responsibilities.

## Scientific philosophy

BMD Agent is physics-first.

The scientific foundation of the group is computational materials
physics using tools including:

- VASP
- pymatgen
- Materials Project
- atomate2
- jobflow
- high-performance computing

AI and machine learning are used to accelerate and enhance this
scientific workflow, not to replace physically meaningful calculations
or scientific validation.

The agent must preserve distinctions between different forms of
evidence. For example:

- an ML prediction is not a DFT result;
- a DFT result is not experimental validation;
- negative formation energy is not equivalent to thermodynamic
  stability against competing phases;
- implementation of a workflow is not evidence that the workflow has
  been validated;
- absence of a composition from a database does not by itself establish
  structural novelty.

Scientific conclusions should remain traceable to their methods,
assumptions, data, and validation state.

## Initial scope

BMD Agent v0 is observational.

It may inspect authorized resources, perform analysis, and provide
evidence-backed scientific and technical advice.

It may not modify authoritative BMD resources.

Initial resources include:

- the local BMD Compute Git repository;
- the local BMDex Git repository;
- the BMD Lab website;
- authorized BMD files on PowerSLURM;
- the BMD PowerSLURM queue;
- VASP calculation outputs;
- pymatgen;
- Materials Project.

BMDwiki and additional scientific tools will be integrated incrementally.

## Governance

BMD Agent is a group resource and a research/training project.

Group members may use the agent and may contribute to its development.

Authoritative changes to BMD resources remain under PI control.

Future capabilities that can modify repositories, submit or cancel
calculations, deploy software, publish information, or adopt scientific
standards must cross an explicit approval boundary.

BMD Agent v0 avoids this problem entirely by not exposing such action
capabilities.

See `ARCHITECTURE.md` and `SECURITY.md` for the architectural and
security contracts.

## Development philosophy

The initial implementation should remain:

- small;
- understandable;
- auditable;
- modular;
- evidence-oriented;
- physics-grounded.

Existing scientific and software tools should be orchestrated rather
than reimplemented.

The project will grow incrementally after each capability has been
tested and validated.
