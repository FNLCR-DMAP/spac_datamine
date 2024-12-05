# GPU Acceleration

## Grapheno Testing Results:
*Fall 2024: Updated December 5, 2024*
1. We attempted to use the original version of Rapids (22.08) Grapheno was implemented in, however there was an issue with the versions
    * pip installing Grapheno was also problematic, so we copied the code over into a file

2. We updated Rapids to the most recent version (24.10), which resolved the issue, however we ran into a 0 worker error with dask.
    * It temporarily resolves itself and the code runs, no apparent reason why

3. Running Grapheno on Anvil was fairly slow (4M cells with 20 features took 66 minutes), which was unexpected, so we moved to the collab implementation

4. Collab implementation is faster, but doesn't utilize dask, so no multicore functionality