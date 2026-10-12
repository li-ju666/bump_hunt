# One model for a bump hunt

Simulation-based inference is normally used where the likelihood is intractable, which means the answers it produces are hard to check. This post goes the other way. It takes a small version of a collider search, written so that the likelihood is available in closed form, and trains a single masked flow matching model, [OneFlowSBI](https://arxiv.org/abs/2601.22951), to answer the questions a search asks: the posterior, the likelihood, the posterior with part of the detector missing, and a score for data outside what the simulator produces. Every answer is compared with the exact one. The results section is interactive.

The toy keeps the features that make searches awkward for learned inference: a random number of events, a narrow signal on a background that falls over two orders of magnitude, a nuisance parameter, a selection cut, and a histogram as the observation.

Contents: [the physics](#physics) · [the simulation](#simulation) · [exact likelihood](#likelihood) · [the method](#method) · [training](#training) · [results](#results) · [open problems](#limits) · [reproduce](#reproduce)

## The physics {#physics}

The LHC collides protons. Each collision is an event, and occasionally its energy turns into a pair of muons, one positive and one negative. Muons leave clean tracks and their momenta are measured well, which makes dimuon events a standard place to look for new particles.

From the two muons' energies and momenta, summed, one computes the invariant mass of the pair,

$$ m = \sqrt{E^2 - |\mathbf{p}|^2}, $$

in units with $c = 1$. This is the mass of whatever decayed into the pair, and it is independent of how fast that parent was moving: the extra energy of a moving parent appears as momentum, and the formula removes it. Each event yields one value of $m$; the data of a search is the histogram of $m$ over millions of events.

Most pairs come from the Drell-Yan process, in which a quark and an antiquark annihilate into a virtual photon or $Z$ boson that then turns into the pair. A virtual particle has no fixed mass, so these pairs occur at every $m$, with decreasing frequency as $m$ grows. They form a smooth, steeply falling spectrum: the background.

A new heavy particle, generically a $Z'$, has a single mass $M$, so its decays add a narrow excess of events near $m = M$ on top of the smooth spectrum. The analysis is called a bump hunt for this reason, and it asks three things: whether there is a bump, where it is, and how large it is.

<!-- mechanism -->

Three features turn this into a statistical problem. A single pair cannot be attributed to either source, since a Drell-Yan pair at 600 GeV is indistinguishable from a $Z'$ pair at 600 GeV, so the signal appears only as an excess. The bump has a width, from the particle's natural line width and from the detector's one to two percent momentum resolution. And the background shape and the detector calibration are known only approximately; these systematic uncertainties enter the model as nuisance parameters, and a wrong background shape can imitate or mask a bump.

## The simulation {#simulation}

The toy has three parameters, $\theta = (\mu, M, \lambda)$. A more detailed version, with a Breit-Wigner line shape, explicit detector smearing and a calibration nuisance, gives the same picture at the cost of a likelihood that needs numerical integration, so the simpler one is used here.

| Symbol | Meaning | Prior |
|---|---|---|
| $\mu$ | signal strength: the number of $Z'$ events relative to a nominal prediction, with $\mu = 0$ meaning no new particle | $\mathcal{U}(0, 3)$ |
| $M$ | $Z'$ mass in GeV | $\mathcal{U}(300, 1000)$ |
| $\lambda$ | decay scale of the background in GeV; a nuisance parameter | $\mathcal{N}(150, 15^2)$ |

The Gaussian prior on a positive quantity follows the convention of these analyses, where the constraint comes from a control measurement with a central value and an error. The probability of a negative draw is below $10^{-23}$.

Constants: $B = 2000$ expected background events, $S = 50$ expected signal events at $\mu = 1$, a generation threshold $m_{\mathrm{gen}} = 200$ GeV, an analysis window of $[250, 1200)$ GeV, a relative peak width $w = 0.02$, and $K = 48$ log-spaced bins with edges $e_0 < \dots < e_K$. One observation is generated in four steps.

1. Event counts: $N_b \sim \mathrm{Poisson}(B)$ and $N_s \sim \mathrm{Poisson}(\mu S)$. Each collision is an independent trial with a small probability of producing a pair, which gives the Poisson limit.
2. Masses: background $m = m_{\mathrm{gen}} + \mathrm{Exponential}(\text{mean } \lambda)$, the simplest falling spectrum; signal $m \sim \mathcal{N}(M, (wM)^2)$, a fixed mass seen through a detector whose resolution scales with mass. Line width and smearing are combined in $w$.
3. Selection: events outside the window are discarded. The number of survivors is random.
4. Histogram: $x_k = \#\{e_{k-1} \le m < e_k\}$. Log-spaced bins have constant relative width, matching the resolution.

$M$ is random in two distinct senses. Within one observation it is a fixed number, and the spread of signal masses around it is physical. Across a training set, $\theta$ is drawn from the prior, which expresses ignorance of the true values. Each training pair is a separate hypothetical universe with its own $\theta$ and a histogram of roughly 1500 events.

The flagging test in the results needs data that the model has no way to explain. For that, a second simulator replaces the exponential background with a power law of density $\propto m^{-4}$ and leaves everything else unchanged. No value of $\theta$ reproduces its spectra, so a model trained on the first simulator should assign them low probability.

## Exact likelihood {#likelihood}

A Poisson number of independent draws, each landing in bin $k$ with some probability, yields independent Poisson bin counts. This is Poisson thinning: the randomness of the total removes the negative correlation that a fixed total would induce. Events outside the window form one more bin that is never observed. With $F_\lambda$ the CDF of the background mass and $\Phi$ the standard normal CDF,

$$ x_k \sim \mathrm{Poisson}(\nu_k), \qquad \nu_k(\theta) = B\,\big[F_\lambda(e_k) - F_\lambda(e_{k-1})\big] + \mu S \left[\Phi\!\Big(\tfrac{e_k - M}{wM}\Big) - \Phi\!\Big(\tfrac{e_{k-1} - M}{wM}\Big)\right], $$

$$ \log p(x \mid \theta) = \sum_{k=1}^{K} \big[\, x_k \log \nu_k - \nu_k - \log x_k! \,\big]. $$

From this expression every reference below follows by summation over a grid in $(\mu, M, \lambda)$: the posterior and its $(\mu, M)$ marginal, the posterior given a subset of the bins (the others are dropped from the sum), the evidence $p(x) = \int p(x \mid \theta)\,\pi(\theta)\,d\theta$, and the likelihood ratio against $\mu = 0$. In an actual search none of these are available, because the probability of an event landing in a bin passes through a full simulation of the collision and the detector.

## The method {#method}

Standard SBI trains one network per query. OneFlowSBI trains one network on the joint distribution of parameters and observations and encodes the query as an input mask. The joint state is

$$ \mathbf{z} = (\mu, M, \lambda, x_1, \dots, x_{48}) \in \mathbb{R}^{51}. $$

A ternary mask $\mathbf{m} \in \{-1, 0, 1\}^{51}$ assigns each coordinate a role: observed (1, held at its value), generated (0, transported from Gaussian noise to the target), or absent ($-1$, excluded from the flow, hence marginalised). The model is a velocity field $\mathbf{v}^\phi_t(\mathbf{z}_t, \mathbf{m})$ on the masked linear interpolant

$$ \mathbf{z}_t = \mathbf{m}^{\mathrm{obs}} \odot \mathbf{z}_1 + \mathbf{m}^{\mathrm{gen}} \odot \big[(1-t)\,\mathbf{z}_0 + t\,\mathbf{z}_1\big], \qquad \mathbf{z}_0 \sim \mathcal{N}(0, I),\ \mathbf{z}_1 \sim p(\theta)\,p(x \mid \theta), $$

trained by regression onto $\mathbf{z}_1 - \mathbf{z}_0$ on the generated coordinates,

$$ \mathcal{L}(\phi) = \mathbb{E}_{t, \mathbf{z}_0, \mathbf{z}_1, \mathbf{m}} \left[ \frac{1}{n_{\mathrm{gen}}} \big\| \mathbf{m}^{\mathrm{gen}} \odot \big( \mathbf{v}^\phi_t(\mathbf{z}_t, \mathbf{m}) - (\mathbf{z}_1 - \mathbf{z}_0) \big) \big\|^2 \right]. $$

At the population optimum, integrating the ODE from $t = 0$ to $1$ on the generated coordinates samples $p(\mathbf{z}_G \mid \mathbf{z}_O)$ with the absent coordinates integrated out, for every mask with positive training probability. The same ODE run backwards with its divergence gives the log density of the generated block. The queries used here, with their masks and references:

| Query | Mask on the joint state $(\mu, M, \lambda;\ x_1 \dots x_{48})$ | Exact reference |
|---|---|---|
| posterior | gen, gen, gen; all obs | grid posterior |
| posterior, nuisance marginalised | gen, gen, absent; all obs | grid posterior, $\lambda$ integrated |
| posterior with a dead detector region | gen, gen, absent; obs except a window of absent bins | grid posterior on the remaining bins |
| likelihood | obs, obs, obs; all gen | the simulator, and the Poisson formula |
| evidence, for flagging | absent, absent, absent; all obs, density read out | grid evidence |
| null predictive | obs, absent, absent; all gen | the simulator at $\mu = 0$ |

Two of these queries are specific to searches. The nuisance is an explicit coordinate, so marking it absent returns the marginal posterior directly rather than by generating and discarding. A dead detector region is a set of absent bins and needs no retraining.

## Training {#training}

The training set is 30,000 pairs $(\theta, x)$ from the prior and the simulator, with a tenth held out for validation.

**Preprocessing.** The flow needs continuous data of roughly unit scale on every coordinate, and raw histograms are neither. Counts are integers, so they are dequantised with fresh uniform noise in $[0, 1)$ on every pass; this gives each integer a unit cube of density and the floor undoes it. Poisson fluctuations grow with the mean, so the first bins would dominate the loss and the nearly empty high-mass bins, where a heavy $Z'$ would show, would barely register; a $\log(1 + \tilde{x})$ transform makes the fluctuations comparable across bins. Then every coordinate, bins and parameters alike, is standardised so that the target sits at the scale of the Gaussian base distribution and the velocities to be learned are of order one. All three steps are invertible with known Jacobians, which is what allows learned densities to be compared with exact ones in count space.

**Masking.** The model answers only the queries it was trained on, so every mask that will be asked for must have positive probability during training. Each example therefore draws its own mask from a mixture: the joint, the posterior, the likelihood, the two marginals, and random partial masks in which each coordinate gets an independent role, roughly in the proportions of the OneFlowSBI paper. Random roles almost never produce a contiguous run of absent bins or the exact pattern of a nuisance-only marginal, so a further family of structured masks covers the physics queries directly: a window of absent bins, the posterior given $\lambda$ or with $\lambda$ absent, and the null predictive. The mixture weights are a trade-off; whatever gets more probability is learned better, and the likelihood, with 48 generated coordinates, is the query that suffers first when it gets too little. Applying a mask is simple: observed coordinates keep their values, generated ones are placed on the interpolant, absent ones are zeroed, and the mask itself is appended to the input, one-hot, so the network knows which case it is in.

**Architecture.** The observation is a short vector with no structure worth a special backbone, so the network is a residual MLP. Two choices matter more than its size. The mask enters as an input alongside the state rather than through per-coordinate tokens, which is what keeps the cost linear in the number of coordinates and lets the same network serve every query. And the flow time $t$ does not enter with the state; it is embedded and turned into a scale and a shift that modulate the normalisation inside each block, so the network can change its behaviour along the flow without the time competing with the data in the first layer. Output layers start at zero, so training begins from the identity flow.

<!-- architecture -->

**Training.** The optimiser and schedule are the standard ones for flow matching and were not tuned. A validation loss on the held-out pairs is used for early stopping. The training takes a few minutes on a Mac Mini M4.

Sampling integrates the masked ODE with 20 Euler steps; the learned paths are close to straight, so finer schemes change nothing visible. Log densities, which need the divergence along the path, use a finer Runge-Kutta integration with the exact divergence, which is cheap at this dimension.

## Results {#results}

All panels use one trained network and the exact references. Blue is exact, orange is the model. Each choice of true signal strength and mass is a separate simulated observation at $\lambda = 150$ GeV. The network does not run in the browser; the twelve scenarios and the missing-bin variants are precomputed.

<!-- explorer -->

The observation panel shows the counts with $\sqrt{x_k}$ error bars, the expected counts $\nu_k$ at the truth, and the background-only expectation dashed. At $\mu = 0$ the two curves coincide; at $\mu = 2$ the bump is about a hundred events in three bins.

The posterior panel shows 800 samples from the network with $\lambda$ absent, over the exact $p(\mu, M \mid x)$ with $\lambda$ integrated out. At $\mu = 1$ the samples follow the contours. At $\mu = 0$ the mass is unidentified and both distributions spread over the prior; the exact posterior shows faint ridges where background fluctuations resemble weak bumps.

"Observed bins" selects the missing-data query. A missing bin is one whose count is treated as unknown, the analogue of a dead detector region, and the model handles it with the absent role. *Off-peak window missing* hides eight bins well below the true mass (or above it, when the mass is near the lower edge of the window); they constrain only the background, so the posterior changes little. *Peak window missing* hides the eight bins centred on the true mass, so the bump is invisible and the posterior on $M$ widens; the exact answer drops those bins from the likelihood and is typically multimodal. The spread of $M$ in the strip above the panels tracks this.

The likelihood panel uses the same network with $\theta$ observed and $x$ generated: a thousand histograms generated at the true parameters against a thousand from the simulator. The bands agree bin by bin, including the nearly empty bins at high mass.

In the flagging panel nothing is generated. The network scores histograms with all three parameters absent, so the score is $\log p(x)$, the evidence. Histograms from the training simulator score around $-140$, histograms from the power-law simulator around $-190$. The threshold slider trades missed shifts against false alarms, and the toggle shows the same experiment with the exact evidence, which the learned scores track. The model returns a posterior for any histogram; this score indicates when that posterior should not be trusted.

## Open problems {#limits}

A fuller evaluation of the same model against the exact references, on observations drawn from the prior, leaves three things worth knowing. Marginalising $\lambda$ with the absent role and marginalising it by generating and discarding give the same distribution; the theory promises this at the optimum, but nothing enforces it after finite training, so it has to be checked. The posterior on $M$ comes out wider than the exact one, whose width is a few GeV, under one percent of the prior range, and mapping a unit Gaussian onto something that narrow is where a flow is weakest; more simulations narrow it. And the learned likelihood ratio against $\mu = 0$ is monotone in the exact one but compressed: when a large bump is present, the exact likelihood under $\mu = 0$ falls by hundreds of nats into a region without training data, and the learned density does not fall as far.

The multimodal cases, with the peak bins absent or $\mu$ near zero, are where the learned posterior is least accurate: it covers the modes but blurs them. Whether this is a budget effect or a property of the straight-line interpolant is unresolved.

The observation here is a fixed-length histogram and the backbone an MLP. Real searches have event-level data of variable size, and the structure to respect is permutation invariance over events rather than a grid of bins. A convolution over bins is the small step; a set-structured backbone is the relevant one.

The budget question is sharper than the number 30,000 suggests. A real simulation is priced per event, each histogram has about 1,500 of them, and searches build histograms from one large background sample and one signal sample per mass point. The accounting for collider SBI should be in events, and the mass region where the posterior is still wide is where additional events would be worth most. That is the direction this toy is meant to prepare.

## Reproduce {#reproduce}

The code, the trained model and the source of this page are at [github.com/li-ju666/bump_hunt](https://github.com/li-ju666/bump_hunt). Training the model takes a few minutes on a laptop; the README there lists the commands.

<p id="model-note"></p>

## References {#references}

1. M. Nautiyal, L. Ju, et al. *OneFlowSBI: One Model, Many Queries for Simulation-Based Inference.* [arXiv:2601.22951](https://arxiv.org/abs/2601.22951), 2026.
2. Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, M. Le. *Flow Matching for Generative Modeling.* ICLR 2023.
3. K. Cranmer, J. Brehmer, G. Louppe. *The frontier of simulation-based inference.* PNAS 117 (48), 2020.
