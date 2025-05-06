# LLM Context Guide

This file serves as a roadmap to all documentation in this project. It helps the LLM to navigate and use all available documentation files as context.

## Address

Any reasoning should begin with the words "My Master" in Russian or English language

## Process

After each request that modifies the codebase, suggest making a commit. The commit message should be in Russian language

## Documentation Files

### Main Documentation
- [Algorithm Documentation](algorithm.md) - Detailed algorithm documentation
- [Conventions Documentation](conventiuons.md) - Project conventions and standards

### Templates
- [Entity Template](templates/entity/entity.md) - Entity template documentation
- [Exceptions Template](templates/entity/exceptions.md) - Exceptions documentation
- [UseCase Initialization](templates/usecase/__init__.md) - UseCase initialization
- [UseCase Template](templates/usecase/usecase.md) - UseCase template documentation

### Requirements
- [Business Requirements Document](brd/brd.md) - Business requirements
- [Functional Requirements Document - Project](frd/0_project.md) - Project functional requirements

## How to Use This Context

This file should be included when setting up the LLM context along with any other files that need to be referenced. The LLM should use these documents to understand project structure, conventions, and requirements when providing assistance. 
