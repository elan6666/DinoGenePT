# Workflow Specification

## Core journey

1. Read README and copy the server setup commands.
2. Run `genept-seed doctor` to check Python, paths, endpoint, and credential
   presence without printing the credential.
3. Run `genept-seed data prepare` to download, verify, and extract official
   inputs.
4. Run `genept-seed embed generate` to checkpoint GenePT-Seed vectors.
5. Run `genept-seed benchmark run` for selected tasks.
6. Inspect compact result and provenance files.

## States

- Empty: explain the next exact command and missing path.
- Running: report completed/total genes or tasks without secret-bearing request
  bodies.
- Error: preserve checkpoint, identify retryable versus terminal failures, and
  exit non-zero.
- Success: print artifact paths, counts, dimensions, and checksums.

There is no graphical interface in v0.

