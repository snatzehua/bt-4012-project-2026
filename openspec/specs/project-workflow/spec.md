## Purpose

Define the repository's current planning and delivery requirements so research changes remain traceable and reproducible.

## Requirements

### Requirement: Traceable research changes
Substantial research or pipeline changes SHALL use an OpenSpec change containing a proposal, requirements, design and an ordered task checklist. Routine editorial corrections MAY be made directly.

#### Scenario: Plan a new experiment
- **WHEN** work changes data selection, preprocessing, modelling or evaluation
- **THEN** the change records its motivation, acceptance criteria, design and tasks before implementation

### Requirement: Honest completion status
The repository SHALL distinguish planning artifacts from completed work. A task SHALL be marked complete only when its output exists and has been checked, and main specs SHALL be synchronized with completed changes.

#### Scenario: Planning files exist but the protocol is unwritten
- **WHEN** a change has proposal, specs, design and tasks files but no required protocol output
- **THEN** the tasks remain incomplete and documentation describes the protocol as planned

### Requirement: Current research sources
Project guidance SHALL use the revised BAF proposal and lecturer feedback as current context, distinguish unresolved choices from decisions, and label archived alternatives as historical.

#### Scenario: Conflicting proposal versions
- **WHEN** an archived proposal differs from the revised proposal or feedback
- **THEN** the current plan cites the revised sources and explains any proposed scope adjustment

### Requirement: Reproducible and acknowledged deliverables
Modelling changes SHALL provide concrete reproduction instructions and record data selection, splits, seeds and dependencies without committing raw datasets. Submitted work SHALL acknowledge AI assistance and external code. The report SHALL follow the course structure and 8-12 page limit excluding references and appendices.

#### Scenario: Add the first modelling pipeline
- **WHEN** runnable modelling code is added
- **THEN** README contains environment, data access and execution instructions, raw data stays out of git, and assistance sources are recorded for the final acknowledgment
