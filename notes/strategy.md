# Localise Measurement Strategy

For a **N = Nx x Ny** grid, (requiring log_2(Nx)+log_2(Ny) qubits) where you need **n individual probabilities**, the most optimal approach is a **Hybrid Amplituse Amplification-Sampling Strategy**.

Running **n** separate Quantum Amplitude Estimations (QAE) is technically precise but computationally expensive. Instead, you can "zoom in" on your region of interest and then sample the internal distribution.

---

## What to do

1. Create transport circuit
2. Create Region Oracle
3. By Iterative Amplitute Estimation (IAE) to find probaility of target region **p**
4. Based on **p** (to select number of iterations for AA), run Amplitude Amplification (AA) to boost the probability of the target region
5. Run Sampling to measure the target region
6. Scale sampling results by **p** to get final individual probabilities

---

## The Strategy: "Amplify, then Sample"

If your **n** states have very low amplitudes compared to the rest of the N-state grid, a regular measurement will waste shots on states you don't care about. The optimal workflow is:

1. **Build a Region Oracle:** Mark the entire **n** block as "good."
2. **Region Probability Estimation:** Use one single QAE run to find the *total* probability of the regio, then estimate **k** (number of iterations) to amplify the region.
3. **Amplitude Magnification (Amplitude Amplification):** Run a **k** iterations of Amplitude Amplification algorithm to "boost" the total probability of that **n** region until it dominates the state (ideally close to 100%).
4. **Measurement (Sampling):** Measure the system several hundred times. Because the region is amplified, almost every "shot" will result in one of your **n** target states.
5. **Scale Sampling Results:** Scale the sampling results by the region probability to get the final individual probabilities.

___

## Iterative Amplitude Estimation (IAE) 
- https://www.qrisp.eu/reference/Primitives/IQAE.html
- Accelerated Quantum Amplitude Estimation (IQAE). This function performs QAE with a fraction of the quantum resources of the well-known QAE algorithm (https://www.qrisp.eu/reference/Primitives/QAE.html).

## Amplitude Amplification (AA) 
- https://www.qrisp.eu/reference/Primitives/amplitude_amplification.html

## Multi Measurement
- https://www.qrisp.eu/reference/generated/qrisp.multi_measurement.html

---

## State preparation
- https://www.qrisp.eu/reference/Primitives/prepare.html
