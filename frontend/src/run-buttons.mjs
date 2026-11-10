// The run's buttons as the run page draws them (main.js renderRun): which of Start, Resume, Step and Pause a press makes
// work now. Words elsewhere name a button only while pressing it works (the stage's note at frame 0, Go deeper's first
// iteration, the knob guide's timing and its Try message): they read this, never the run's state on their own.

/** {start, resume, step, pause}: whether a press of each works now. `run` the run's record (its state and proseg's
 *  ack), `frames` the frames the page holds (one alone: the run is still on its prior, where the run button says Start),
 *  `prompt` the finish prompt is open (the run button and Step wait for it). The run button is one of Start, Resume and
 *  Pause; it waits while proseg pauses or resumes (its ack lags the run's state). */
export function runButtons(run, frames, { prompt = false } = {}) {
  const paused = run?.state === 'paused' && run.ack?.state === 'paused';
  const pausing = run?.state === 'paused' && !paused;
  const resuming = run?.state === 'running' && run.ack?.state === 'paused';
  const sampling = ['running', 'paused'].includes(run?.state);
  const toggle = sampling && !prompt && !pausing && !resuming;
  const onPrior = paused && frames?.length === 1;
  return { start: toggle && onPrior, resume: toggle && paused && !onPrior, step: paused && !prompt, pause: toggle && !paused };
}
