# Forgetting might be one of the keys to generalization 🧠

The concept of "forgetting" is nothing new in machine learning. From the forget gates in LSTMs decades ago, to standard regularization techniques like dropout and weight pruning, we've long known that models shouldn't memorize everything. 

However, I wanted to run a specific architectural experiment: what happens if we apply a highly dynamic, biologically-inspired "use-it-or-lose-it" mechanism directly to the activations of a modern SSM-Transformer during pre-training?

This repository explores an experimental module called the **Cognitive Forgetting Gate**. The goal was to see if we could force the model to naturally separate "rigid grammar rules" from "flexible context" by making it continuously earn its neural pathways. 

Here is what the experiment looked like, the failure modes I ran into, and the ultimately encouraging results.

---

## 🔬 The Setup: A Dynamic Regularizer
In standard training, a model might dedicate capacity to memorizing noise or specific quirks of the training data. To counter this, I added a gate inside the MLP layers that tracks how consistently a neuron fires over time using an Exponential Moving Average.

1. **Decay (Forgetting):** If a neuron fires erratically or rarely, its "health" slowly decays. Its influence is dampened.
2. **Consolidation (Locking):** If a neuron fires consistently over thousands of steps—proving it represents a fundamental rule—it hits maximum health and "locks." It is then permanently shielded from decay.

---

## 🚧 The Journey & Observations

### Observation 1: The "Islands" Problem (When forgetting is too aggressive)
My first iteration of the gate was far too aggressive. It looked at the complex, fragile pathways that connect concepts together (like connecting the word "grandfather clock" to "midnight") and, because they fired rarely, it killed them off.

Meanwhile, it saw the neurons responsible for basic punctuation and dialogue tags firing constantly, and permanently locked them in. 

**The Result:** The model turned into a collection of perfect grammatical "Islands" with zero "Bridges". It produced structurally flawless sentences, but had no attention span. It would jump from talking about a clock, to a knight in armor, to a philosophical quote about sheep—all in three sentences. 

*Lesson learned: You cannot prune connections before the model has had time to learn the basics. A model needs a "warmup period" and a baseline level of plasticity to function.*

### Observation 2: Rebuilding the Bridges
To fix the context-switching, I updated the architecture with a more balanced approach:
1. **The Health Floor:** Unused neurons no longer drop to 0% health; they just "go to sleep" (dropping to ~20%). This allows them to wake back up when a specific, rare topic is mentioned in the prompt.
2. **Wildcard Neurons:** I left 25% of the neurons completely ungated. They are never allowed to forget. These act as a highly plastic reserve to maintain long-term context.
3. **Layer-Specific Decay:** Early layers (which tend to handle rigid grammar) were set to forget aggressively. Deeper layers (which handle abstract logic) were kept highly flexible.

### Observation 3: The Sweet Spot
Once the network was allowed to maintain these "bridges," the training stabilized beautifully. The metrics settled into a very healthy equilibrium:
* **~16%** of the gated neurons permanently locked in (crystallizing the rigid rules of language).
* **~84%** of the gated neurons dynamically fluctuated in health, going to sleep and waking up based on the current topic.
* **100%** of the Wildcard neurons remained active to carry the context.

The model stopped randomly jumping between topics and started demonstrating "Semantic Drift"—smoothly connecting related ideas. Here is an actual, unedited output from the model during training when prompted about government grants:

> *- The "Exit from Thermal Sieve" bonus, with a maximum value of €1,500, which concerns thermal sieves classified in category E ;*
> *- The "Low Consumption Building" bonus, for homes that achieve energy label A or B.*
> *- The last bonus concerns "Comprehensive Renovation" with a maximum value of €7,000...*
> 
> *As part of Energy Savings Certificates (CEE), energy suppliers or fuel sellers have launched the Prime Énergie Monexpert. This covers thermal renovation works for households based on income conditions... Moreover, wind turbines are able to generate more than 100 gigawatts of power through their photovoltaics... To learn more about our own ESG portfolio, please check out these resources: `https://www.watercounselproject.org/`*

Notice the execution of bullet points, the logical transition from French thermal grants to renewable energy, and the generation of a highly plausible (but hallucinated) URL at the end. For an experimental model mid-training, this was a great result.

---

## 💡 Key Takeaways

While this is just an experiment, it reinforced a few interesting concepts about network architecture:

1. **Forgetting is a powerful structural tool.** By forcing the network to let go of weakly-firing pathways, you can strip away noise and leave behind highly confident grammatical structures.
2. **Total plasticity isn't always efficient.** Locking in fundamental rules (like syntax) early in the network seems to free up the rest of the model's capacity to focus purely on context and logic. 
3. **Balance is everything.** A functional network needs both crystallized memory (habits, grammar) and fluid memory (adapting to new situations). Reserving a chunk of ungated, highly plastic "wildcard" neurons is crucial to keep the model from becoming too rigid.

## Note to self:
Curriculum learning (learn like a human) seems to be a necessity otherwise the model learns well useless stuff like school bus standards, pesticides or yoga :D!!!
