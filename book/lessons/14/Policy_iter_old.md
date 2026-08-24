# Lesson 14 — Reinforcement Learning II (Policy Iteration)

:::{admonition} Lesson Objectives
:class: note

* Explain and define policy extraction, policy evaluation, and policy iteration.
* Apply policy iteration to a simple decision problem.
* Compare policy iteration and value iteration.
* Explain relationship to reinforcement learning.
:::

## Introduction to Policy Operations

In the previous lesson, we used Value Iteration to calculate the optimal numerical utility of every state on a map. However, our tactical agents ultimately need a playbook—a policy ($\pi$)—to tell them exactly which action to execute. 

To bridge the gap between mathematical state-values and actionable commands, we rely on three distinct operational concepts: **Policy Extraction**, **Policy Evaluation**, and **Policy Iteration**. All three of these algorithms are variations of the core Bellman updates, relying on fragments of one-step expectimax lookaheads. They differ only in whether we plug in a fixed policy or maximize over all possible actions.

### 1. Policy Evaluation
Sometimes an AI agent is simply handed a standard operating procedure (a fixed policy) and needs to calculate how effective it is. **Policy Evaluation** takes a specific, fixed policy and computes the state values strictly for that chosen policy. 

Because the agent's actions are already decided by the playbook, we remove the $\max_a$ operator from the Bellman equation, turning the complex search into a straightforward calculation:

$$V_{k+1}^\pi(s) \leftarrow \sum_{s'} T(s, \pi(s), s') [R(s, \pi(s), s') + \gamma V_k^\pi(s')]$$

### 2. Policy Extraction
Once an agent knows the optimal values ($V^*$) or the evaluated values of a specific policy ($V^\pi$), it needs to determine its next move. Because deciding how to act based solely on state-values is not obvious, the agent must perform a mini-expectimax search (a one-step lookahead) to find the best immediate action. 

This process of converting state-values into an actionable playbook is called **Policy Extraction**. It utilizes the $\arg\max_a$ operator, which returns the actual action that yields the highest score, rather than the score itself:

$$\pi^*(s) = \arg\max_a \sum_{s'} T(s,a,s') [R(s,a,s') + \gamma V^*(s')]$$

### 3. Policy Iteration
**Policy Iteration** is an alternative algorithm to Value Iteration that directly searches for the optimal playbook. The algorithm asks a simple question: *"I have picked a policy; is it optimal?"*. 

It answers this by looping through two steps:
1. Evaluate the current policy using Policy Evaluation.
2. Pick the next policy using Policy Extraction to see if a mathematically better move exists. 

If the newly extracted policy is identical to the old one, the algorithm knows it has found the optimal solution and exits. Mathematically, generating the new policy looks like this[cite: 9]:

$$\pi_{i+1}(s) = \arg\max_a \sum_{s'} T(s,a,s') [R(s,a,s') + \gamma V^{\pi_i}(s')]$$

## Applying Policy Iteration

Let us apply this to a simple tactical scenario involving an autonomous sentry deciding whether to patrol or recharge.

```{mermaid}
graph LR
    S((Base<br>State)) -- "Patrol" --> P[Reward: +5]
    S -- "Recharge" --> R[Reward: 0]

    classDef default fill:#f9f9f9,stroke:#888,stroke-width:2px,color:#000;
    classDef base fill:#d4edda,stroke:#28a745,stroke-width:2px,color:#000;
    linkStyle default stroke:#a0a0a0,stroke-width:2px;
    class S base;