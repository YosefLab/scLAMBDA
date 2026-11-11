# Global review (optional)

Run only when the user asks for a whole-section consistency check after local
annotation. During the main loop, the critic can already request wider views.

1. Render each relevant section without a highlighted target:

   ```bash
   lambdakit spatial image --run RUN --no-target --slide SLIDE --number
   lambdakit spatial adjacency --run RUN --slide SLIDE
   ```

2. Look for a visible relation between niches or landmarks that contradicts a
   current label. Naming preferences, depth, or wanting a narrower name are not
   contradictions. Neighbour labels are predictions, not an anatomical
   reference; one unusual neighbour does not make a label impossible, and an
   empty neighbour list may mean neighbours are not annotated yet.
3. For a real contradiction, record it with `spatial observe` on the affected
   niche and rerun that niche's critic with all of its evidence. A new, more
   specific name needs spatial evidence and the original measurements. Run any
   checks it requests while rounds remain; then save with `dossier decision`
   and commit.
4. When every proposed change is handled, record the phase:

   ```bash
   lambdakit phase done --run RUN --name consistency-review
   ```

If the review itself fails, the local labels stand; report that the review did
not complete.
