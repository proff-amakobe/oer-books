# Planning Intelligently

**From Goals to Actions — Classical and Probabilistic Planning**

*CSC5350 · Artificial Intelligence*

---

## Opening Narrative

### The Fourteen-Minute Window

On August 5, 2012, NASA's Curiosity rover entered the Martian atmosphere traveling at 13,000 miles per hour. What followed was fourteen minutes of autonomous action — the time it takes a signal to travel from Mars to Earth and back — during which the rover had to perform a precisely choreographed sequence of maneuvers: deploy a supersonic parachute, jettison its heat shield, fire retrorockets, lower itself on a sky crane, cut the cables, and land safely on the surface of Mars. No human could intervene. The entire sequence had to work perfectly the first time.

The sequence was planned on Earth months before launch. Mission planners spent years working through every contingency: what to do if the parachute deployed late, if wind pushed the rover off course, if a retrorocket underperformed. The plan that ran those fourteen minutes was the product of computational planning algorithms sophisticated enough to reason about thousands of conditional action sequences and verify that each one would achieve the goal under its specified conditions.

Planning is what distinguishes an agent that *reacts* from an agent that *reasons*. A purely reactive agent responds to what it senses now. A planning agent reasons about sequences of actions before taking them — anticipating consequences, accounting for contingencies, and selecting the course of action most likely to achieve its goals.

> **"The difference between a reactive system and an intelligent agent is the ability to imagine a future different from the present, and to choose actions that bring that future about."**

---

## Learning Objectives

After completing this chapter, you will be able to:

1. Define the classical planning problem and express it in the STRIPS formalism.
2. Describe planning as a search problem and explain why domain-specific search strategies outperform general ones.
3. Implement forward state-space search and regression planning and explain when each is appropriate.
4. Describe the Planning Graph and Graphplan algorithm, and explain how mutex relationships accelerate planning.
5. Extend classical planning to handle temporal constraints, resource limitations, and concurrent actions.
6. Formalize sequential decision problems as Markov Decision Processes (MDPs).
7. Apply value iteration and policy iteration to compute optimal policies for MDPs.
8. Explain the relationship between planning and reinforcement learning, and describe when each is appropriate.
9. Build the IAAIS Planner — a component that generates action sequences from goals and integrates with the Knowledge Base.

---

## Key Terminology

| Term | Plain-Language Definition |
|---|---|
| **Planning** | The process of selecting a sequence of actions that, when executed, transforms an initial state into a state satisfying the goal. Distinguished from search by the use of domain-specific representations. |
| **STRIPS** | Stanford Research Institute Problem Solver — the first widely used planning formalism. Represents actions by their preconditions (what must be true before execution) and effects (what becomes true or false after). |
| **Action Schema** | A parameterized template for an action class. Move(?robot, ?from, ?to) describes all possible move actions for all robots, sources, and destinations with a single schema. |
| **Precondition** | The condition that must hold in the current state for an action to be applicable. If a precondition is not satisfied, the action cannot be executed. |
| **Effect** | The change an action makes to the world. Add effects assert new facts; delete effects remove facts that are no longer true. |
| **State Space** | The set of all possible world states the planner can reason about. Planning searches through this space for a path from the initial state to the goal. |
| **Forward Search** | Planning by starting from the initial state and applying applicable actions to generate successor states, searching forward until a goal state is reached. |
| **Regression Planning** | Planning by starting from the goal and identifying which actions could achieve it, working backward to find applicable actions until the initial state is reached. |
| **Planning Graph** | A layered graph alternating between fact layers and action layers, used by Graphplan to efficiently compute heuristics and identify impossible subgoals. |
| **Mutex** | A mutual exclusion relation between two facts (or actions) indicating they cannot both be true (or both be applied) at the same time. Mutex detection prunes impossible plans early. |
| **Heuristic (Planning)** | An admissible estimate of plan length computed from the planning graph or relaxed problem. Key to making planning tractable for large domains. |
| **Temporal Planning** | Planning with actions that have durations, overlapping execution, and time-bounded goals. Required for domains like manufacturing, logistics, and surgical scheduling. |
| **Contingent Plan** | A plan with conditional branches — "if X then do A, else do B" — for handling uncertainty about which actions will succeed or which observations will be made. |
| **Markov Decision Process (MDP)** | A formal model for sequential decision-making under uncertainty. Defines states, actions, transition probabilities, and rewards; the solution is a policy mapping states to actions. |
| **Policy (π)** | A mapping from states to actions — what the agent should do in each situation. The goal of MDP solving is to find the optimal policy. |
| **Value Function (V)** | The expected cumulative reward achievable from each state under a given policy. The optimal value function V* corresponds to the optimal policy. |
| **Bellman Equation** | The recursive equation expressing the value of a state as the immediate reward plus the discounted value of successor states: V*(s) = max_a [R(s,a) + γ Σ T(s,a,s')V*(s')]. |
| **Value Iteration** | An algorithm for computing the optimal value function by iteratively applying the Bellman equation until values converge. |
| **Policy Iteration** | An alternative MDP algorithm: evaluate the current policy, then improve it by acting greedily with respect to the value function. Alternates evaluation and improvement until convergence. |
| **Discount Factor (γ)** | A value in [0,1) weighting future rewards. γ = 0.9 means a reward 10 steps away is worth 0.9^10 ≈ 0.35 of an immediate reward. |

---

## Section 1 — The Planning Problem

Planning is the task of finding a sequence of actions that achieves a goal. This sounds like search — and indeed, planning can be formulated as search through a state space. But planning distinguishes itself from general search in two ways: it uses a *factored* representation of states (as sets of logical facts rather than opaque objects), and it exploits this structure to reason more efficiently than general-purpose search algorithms.

### STRIPS: The Foundational Formalism

STRIPS, developed at SRI in the early 1970s, established the vocabulary still used in modern planning. A **STRIPS problem** has four components:

- **Initial state:** A set of ground atoms true at the start. `{At(robot, A), Battery(full), Holding(nothing)}`
- **Goal:** A set of atoms that must be true in the final state. `{At(robot, C), Holding(package)}`
- **Actions:** Parameterized schemas with preconditions and effects

```
Action: Pick(robot, package, location)
  Precondition:  At(robot, location) ∧ At(package, location) ∧ Holding(nothing)
  Add effects:   Holding(package)
  Delete effects: At(package, location), Holding(nothing)

Action: Move(robot, from, to)
  Precondition:  At(robot, from) ∧ Connected(from, to)
  Add effects:   At(robot, to)
  Delete effects: At(robot, from)

Action: Drop(robot, package, location)
  Precondition:  At(robot, location) ∧ Holding(package)
  Add effects:   At(package, location), Holding(nothing)
  Delete effects: Holding(package)
```

A valid plan is a sequence of action instances whose execution transforms the initial state into a state satisfying the goal. The **validity** of a plan can be mechanically verified: apply each action in sequence, checking preconditions and updating the state, and confirm the goal holds at the end.

The factored representation enables a critical insight: many facts about the world are irrelevant to any particular planning problem. A robot planning to pick up a package in room A does not need to reason about the weather on Mars. Planners exploit this by reasoning only about facts that are relevant to the current goal — a form of goal-directed pruning that general search cannot apply.

---

## Section 2 — Planning as Search

The most direct way to solve a STRIPS problem is to search through the state space: start from the initial state, apply applicable actions, and continue until a goal state is reached.

**Forward state-space search** generates successor states by applying applicable actions. It is complete and, with appropriate heuristics, efficient. The challenge is that many actions may be applicable in each state — the branching factor can be large — and many will lead away from the goal. Good heuristics are essential.

**Regression planning** works backward from the goal: identify which actions could achieve the current goal state, compute which state would need to be true *before* that action (the action's preconditions, with the goal added back), and repeat until the initial state is reached. Regression is often more efficient because it reasons only about facts that are relevant to the goal.

The key insight of modern planning research is that **good heuristics** are what separate tractable from intractable planning. The most effective heuristics come from relaxed problems: remove the delete effects from all actions, solve the resulting problem (which is much easier — nothing ever becomes false), and use the solution length as a lower bound. This "ignore delete lists" heuristic underlies most state-of-the-art planners.

### The Planning Graph and Mutex Relations

The **Planning Graph** (Blum & Furst, 1995) is a compact data structure that alternates between layers of facts and layers of actions. Starting from the initial facts, each action layer contains all actions whose preconditions appear in the preceding fact layer; each fact layer contains all facts achievable by actions in the preceding action layer.

The planning graph also tracks **mutex relations** — pairs of facts or actions that cannot both be achieved simultaneously. Two facts are mutex if they can only be produced by actions that interfere with each other. These mutex relations prune impossible subgoals before any explicit search: if a required goal fact is always mutex with another required goal fact at the same planning graph level, no plan of that length can exist.

---

## Section 3 — Beyond Classical Planning

Classical planning assumes a deterministic world, complete information, and instantaneous actions. Real planning problems rarely satisfy all three.

### Temporal Planning

Real actions take time. Administering a medication takes five minutes; waiting for lab results takes two hours; scheduling a surgery requires a six-hour block. **Temporal planning** extends STRIPS with action durations and temporal constraints, enabling planners to reason about which actions can overlap, when resources become available, and whether time-bounded goals can be achieved.

PDDL (Planning Domain Definition Language) is the standard language for expressing temporal planning problems, used in AI planning competitions that benchmark planning systems against each other on standardized domains.

### Planning Under Uncertainty: Contingent Plans

When actions can fail or observations can vary, the right representation is a **contingent plan** with conditional branches. "Move the robot to room B; if the door is locked, execute the unlock procedure; otherwise, proceed to the package." Contingent plans can be represented as decision trees (small) or policy graphs (large and cyclic), and constructing optimal ones requires reasoning about the probability and consequences of each contingency.

For problems with significant uncertainty, Markov Decision Processes provide a cleaner and more tractable formalism.

---

## Section 4 — Markov Decision Processes: Planning Under Uncertainty

A **Markov Decision Process** models sequential decision-making where actions have probabilistic outcomes. The agent knows the current state, chooses an action, and transitions to a new state with a probability that depends only on the current state and action — the **Markov property**. A reward is received at each step, and the agent's goal is to maximize the expected cumulative discounted reward.

### The MDP Framework

An MDP has four components:
- **State space S:** All possible world configurations
- **Action space A:** All possible agent choices in each state
- **Transition function T(s, a, s'):** P(next state = s' | state = s, action = a)
- **Reward function R(s, a):** Expected immediate reward for action a in state s
- **Discount factor γ:** How much future rewards are valued relative to immediate ones

The goal is to find a **policy** π: S → A that tells the agent what action to take in each state, maximizing the expected cumulative discounted reward.

### Value Iteration: Computing the Optimal Policy

**Value iteration** computes the optimal value function V*(s) — the maximum expected cumulative reward achievable from state s — by repeatedly applying the Bellman optimality equation:

**V_{k+1}(s) = max_a [ R(s,a) + γ Σ_{s'} T(s, a, s') V_k(s') ]**

Starting from arbitrary initial values, this update is applied to every state repeatedly. It can be proven that V_k converges to V* as k → ∞. In practice, convergence is declared when the maximum change across all states falls below a small threshold ε.

Once V* is computed, the optimal policy is extracted greedily:

**π*(s) = argmax_a [ R(s,a) + γ Σ_{s'} T(s, a, s') V*(s') ]**

```
# Value iteration on a medical treatment MDP
# States: {untreated, treated_early, treated_late, recovered, declined}
# Actions: {watch_and_wait, treat_now}

# After convergence of value iteration:
V* = {
  'untreated':     8.2,    # High value — still time to treat
  'treated_early': 9.8,    # Highest — best outcome likely
  'treated_late':  6.1,    # Lower — treatment less effective
  'recovered':    10.0,    # Terminal good outcome
  'declined':      0.0,    # Terminal bad outcome
}

π* = {
  'untreated':     'treat_now',        # Don't wait
  'treated_early': 'continue_treatment',
  'treated_late':  'continue_treatment',
}

# Policy extraction shows: treat as early as possible.
# The value function captures the cost of delay implicitly
# through discounting and transition probabilities.
```

**Policy iteration** is an alternative that often converges faster. It alternates between two steps: **policy evaluation** (compute the value function for the current policy) and **policy improvement** (update the policy to be greedy with respect to the computed values). Policy iteration is guaranteed to converge in a finite number of iterations because there are finitely many deterministic policies.

---

## Section 5 — Planning in the Real World

### Logistics and Supply Chain

Industrial planning systems manage the movement of goods, personnel, and resources across supply chains with thousands of components and tight temporal constraints. Airlines use automated planners to construct crew schedules, gate assignments, and maintenance windows — and to rapidly replan when weather or mechanical issues disrupt the original plan. FedEx and Amazon use planning systems to optimize package routing and delivery sequencing across millions of packages daily.

### Surgical Planning

Modern surgical planning goes beyond scheduling. For complex procedures, planning systems analyze imaging data to identify the safest approach angles, compute the sequence of anatomical structures that must be navigated, and simulate the procedure before execution. Robotic surgery systems use planning to translate a surgeon's high-level intention into precise tool movements, accounting for tissue deformation and instrument kinematics.

### Healthcare Resource Allocation

Hospital administrators face planning problems of genuine complexity: assigning staff to shifts, allocating operating rooms, scheduling patient flows, managing bed occupancy. MDP-based approaches model patient arrivals as stochastic processes and compute policies for resource allocation that minimize waits, maximize throughput, and satisfy staffing constraints. The difference between a hand-crafted schedule and an optimized one can translate into millions of dollars in operational efficiency and measurable improvements in patient outcomes.

---

## Section 6 — IAAIS Integration: The Planner

This week you add the **IAAIS Planner** — a component that generates action sequences from goals and integrates with the Knowledge Base to maintain an up-to-date world model.

The Planner connects directly to the Knowledge Base: it queries the KB for the current state, asks the Search Engine to find action paths, and writes completed plans back to the KB. When actions are executed (by the Decision Agent, Chapter 11), the KB is updated to reflect the new state.

For deterministic planning (known outcomes, complete information), your IAAIS Planner uses forward state-space search with a relaxed-problem heuristic. For planning under uncertainty (stochastic actions, partial observability), it uses value iteration or policy iteration over an MDP formulation of the domain.

| Chapter | Module | Capability |
|---|---|---|
| Ch 2 | Search Engine | Path planning |
| Ch 3 | Knowledge Base | Structured facts and inference |
| Ch 4 | Planner | Goal-directed action sequences |

---

## Hands-On Exploration: Planning for a Hospital Logistics Robot

### The Activity

Open `hands_on_ch4.ipynb` from the course repository. The notebook contains a hospital floor plan with rooms, corridors, supply closets, and patient rooms; a robotic transport system that can move medical supplies between locations; and a set of supply requests that arrive dynamically.

**Part 1 — STRIPS Formulation (15 minutes):** Formulate the supply transport problem in STRIPS. Define the facts (at least 10), action schemas (at least 3), a test initial state, and a test goal. Manually trace a valid plan for the test case.

**Part 2 — Forward Planning (20 minutes):** Implement forward state-space search with two heuristics: (a) number of unsatisfied goal conditions (inadmissible but often effective) and (b) relaxed plan length (admissible). Compare planning time, plan length, and nodes expanded across 5 test scenarios.

**Part 3 — MDP Formulation (20 minutes):** The robot's battery discharges stochastically: each move has a 90% chance of succeeding and a 10% chance of requiring the robot to return to the charging station. Formulate this as an MDP and compute the optimal policy using value iteration. How does the policy change as the battery threshold for "low battery" changes?

### Reflection Questions

1. STRIPS assumes actions always succeed (deterministic) and the world is fully observable. For your hospital robot scenario, identify two ways the real world violates these assumptions and describe how they could be handled by contingent planning or MDPs.
2. The relaxed-problem heuristic ignores delete effects. Why does this produce an *admissible* heuristic? Can you construct an example where this heuristic is tight (close to the true plan length) and one where it is very loose?
3. Your MDP has a discount factor γ. Setting γ = 0.99 (far-sighted) versus γ = 0.5 (myopic) produces different optimal policies. For the hospital robot, which is more appropriate? Justify your answer by describing what each policy would do when the robot has low battery and an urgent supply request.
4. In your IAAIS Planner, how will you handle goals that are partially conflicting — actions that advance one goal while impeding another? What formalism would you use?

---

## Case Study: NASA's Mars Rover — Planning in the Void

### The Autonomy Imperative

Communication delays between Earth and Mars range from 3 to 22 minutes depending on orbital positions — meaning a round-trip signal takes between 6 and 44 minutes. No human operator can supervise a rover in real time. Any command sequence must be planned on Earth, uploaded, and executed autonomously — with the rover detecting and recovering from problems without human intervention.

NASA's MAPGEN (Mixed-initiative Activity Planning Generator) system, used for Mars rover operations since the Spirit and Opportunity missions, is one of the most consequential AI planning systems ever deployed. Scientists upload daily science objectives — what observations they want, which rocks to analyze, which images to capture. MAPGEN generates a 24-hour activity plan respecting power budgets, thermal constraints, communication windows, and equipment limitations.

### The Challenges

Planning for Mars rovers revealed challenges that do not arise in textbook problems. Actions have durations and consume resources, requiring temporal planning rather than classical planning. The terrain is uncertain — wheels may slip, rocks may be harder than expected — requiring contingent branches and recovery procedures. Power is severely limited, requiring explicit resource accounting across the entire plan.

Perhaps most importantly, scientists have conflicting goals: every researcher wants their instruments to have more time, their experiments to run first, their observations to be collected. MAPGEN operates in a **mixed-initiative** setting — it generates candidate plans, scientists inspect and modify them through a GUI, and the system checks that modifications remain feasible. Human expertise and AI planning complement each other rather than compete.

### The Lesson

The Mars rover missions demonstrated something broader than planning efficiency: AI planning can extend human reach into environments where direct human operation is impossible. The constraint is not intelligence but communication latency. As communication delays grow — missions to the asteroid belt, eventual Mars surface operations — the autonomy of the planning systems must grow with them.

This is a pattern that extends beyond space exploration. Autonomous surgical systems, deep-sea robots, and embedded medical devices all face analogous constraints: the environment makes continuous human supervision impractical or impossible, so the system must plan and adapt autonomously within pre-specified boundaries. The planning algorithms in this chapter are the foundation of that autonomy.

---

## Chapter Summary

We began this chapter in the fourteen-minute communication gap between Earth and Mars — a physical constraint that makes autonomous planning not a convenience but a necessity.

STRIPS gave us the foundational vocabulary: initial state, goal, and parameterized actions with preconditions and effects. The factored representation distinguishes planning from general search, enabling goal-directed reasoning about relevant facts rather than opaque state objects.

Forward and regression planning showed how the state space can be searched in both directions, with the choice depending on the problem's structure. The Planning Graph and mutex relations gave us efficient heuristics that make large planning problems tractable.

Extensions to classical planning — temporal durations, resource constraints, contingent branches — addressed the gap between textbook formalism and real deployment. Mars rovers, surgical planners, and logistics systems all require these extensions.

Markov Decision Processes gave us the framework for planning under uncertainty: when actions have probabilistic outcomes, the right concept is a policy — a mapping from states to actions that is optimal in expectation. Value iteration and policy iteration compute these policies by iteratively applying the Bellman equation.

In Chapter 5, we broaden the uncertainty framework to Bayesian reasoning — handling not just stochastic action outcomes but uncertain observations, incomplete information, and probabilistic inference across complex dependency structures.

---

## Discussion Questions

1. **Representation and tractability:** A planning problem with 100 boolean facts has 2^100 possible states — impossible to enumerate. Yet planners routinely handle problems of this scale. What properties of STRIPS representations enable planners to avoid enumerating all states?
2. **The frame problem revisited:** STRIPS handles the frame problem through explicit delete effects — if an action doesn't delete a fact, the fact persists. This is the "closed-world dynamics assumption." In a real hospital scenario, what real-world changes could violate this assumption?
3. **MDP rewards and values:** In the medical treatment MDP from Section 4, the reward function encodes clinical values — what outcomes are good and bad, and by how much. Who should define these rewards? What are the ethical implications of the reward function choice?
4. **Planning vs. RL:** Both planning (MDPs + value iteration) and reinforcement learning (Chapter 11) compute policies for sequential decision problems. What is the fundamental difference between them? When would you use planning rather than RL for your IAAIS system?
5. **Contingent planning and failure:** A robot plan assumes the elevator will work. When the elevator is out of service, the plan fails. Design a contingent plan for a hospital robot that handles elevator failure gracefully. What is the cost (in plan complexity) of adding this contingency?
6. **Temporal constraints and ethics:** A surgical planning system determines that a procedure cannot be completed safely within the available OR time. The right action is to reschedule. But the patient has been waiting months and the next available slot is in six weeks. How should this tradeoff be represented in the planning objective? Who should make the decision?
7. **Multi-agent planning:** Hospital logistics involves multiple robots, multiple staff, multiple patients, and multiple competing goals. What additional challenges arise in multi-agent planning that do not appear in single-agent planning?
8. **Your IAAIS Planner:** For your domain, identify three planning scenarios of increasing complexity: (a) deterministic with complete information, (b) deterministic with incomplete information, (c) stochastic. Describe how you would handle each and what additional capabilities your IAAIS Planner would need for scenario (c).

---

## Further Reading

### Classical Planning

Ghallab, M., Nau, D., & Traverso, P. (2004). *Automated Planning: Theory and Practice*. Morgan Kaufmann. The comprehensive reference for classical and temporal planning.

Blum, A., & Furst, M. L. (1997). Fast planning through planning graph analysis. *Artificial Intelligence*, 90(1–2), 281–300. The Graphplan paper — planning graphs and mutex reasoning.

### MDPs and Decision Theory

Puterman, M. L. (1994). *Markov Decision Processes: Discrete Stochastic Dynamic Programming*. Wiley. The authoritative mathematical treatment.

Russell, S., & Norvig, P. (2020). *AI: A Modern Approach* (4th ed.). Chapters 16–17. Accessible MDP coverage with planning connections.

### Applications

Chien, S., et al. (2000). Using iterative repair to improve responsiveness of planning and scheduling for autonomous spacecraft. *AIPS 2000*. NASA planning systems in practice.

---

*— End of Chapter 4 —*
